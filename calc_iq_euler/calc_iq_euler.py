#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 02:12:29 2022

@author: zijiewu
"""
import numpy as np
import numexpr as ne
import time
import math
#%%
def fibonacci_sphere(samples=1000):

    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return np.array(points)
def calc_iq_euler_np(qrange,rs,fis,total_points = 100):
       
    rvs = []
    ret = np.zeros(len(qrange))
    v_array = fibonacci_sphere(total_points).transpose()
#    i = 0
#    for theta in np.linspace(0,1,theta_spacing):
#        v_theta = np.arccos(2*theta-1)
#        for phi in np.linspace(0,1,phi_spacing):
#        
#            v_phi = 2*np.pi*phi
#        #convert from spherical to cartesian
#            v_array[:,i] = np.array([np.cos(v_phi)*np.sin(v_theta),
#                          np.sin(v_phi)*np.sin(v_theta),
#                          np.cos(v_theta)])
#            i += 1
         
    for ri,r in enumerate(rs):
        r_array = np.array(r)
        rvs.append(np.matmul(r_array,v_array))
    
        for qi,q in enumerate(qrange):
            sum_cos = np.zeros(total_points)
            sum_sin = np.zeros(total_points)
            for rvi,rv in enumerate(rvs):
                sum_cos += ne.evaluate("sum(cos(rv*q),axis=0)")*fis[rvi]
                sum_sin += ne.evaluate("sum(sin(rv*q),axis=0)")*fis[rvi]
            sum_cos_sin = sum_cos**2+sum_sin**2
            ret[qi] += np.mean(sum_cos_sin)
    
    return ret                