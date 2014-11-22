#!/usr/bin/env python 
import numpy as np
import os, sys
from math import *
from subprocess import Popen

#Sigmas = {"x":(0.02,0.03),"y":(0.07,0.02),"z":(0.08,0.5),"Phix":(0.0003,0.01),"Phiy":(0.0015,0.027),"Phiz":(0.0004,0.0004)} # (DT,CSC) tuple
#APEs = {"x":([0.01,0.02,0.03],[0.02,0.03,0.04]),"y":([0.05,0.07,0.1],[0.01,0.02,0.03]),"z":([0.05,0.08,0.1],[0.4,0.5,0.6]),"Phix":([0.0001,0.0003,0.0005],[0.008,0.01,0.012]),"Phiy":([0.001,0.0015,0.003],[0.020,0.027,0.032]),"Phiz":([0.0002,0.0004,0.0006],[0.0002,0.0004,0.0006])} # (DT,CSC) tuple of lists

Sigmas = {"xxx":(0.,0.)}
APEs = {"xxx":([0.],[0.])}

input_file = open('template.xml','r').readlines() #everything will be replaced
length = len(input_file)

Scenarios = [0] #[0,1,2]
scenLabel = ['Test'] #['Loose','Medium','Tight']

for scenario in Scenarios:
    for dof in Sigmas.keys():
        output_file = open("artifScenario_Sigma"+dof+'_'+scenLabel[scenario]+".xml",'w') #str(sys.argv[2]),'w')
        for iline in range(length):
            if input_file[iline].find('setape') != -1:
                if dof == "xxx": output_file.write("    <setape xx=\"%4.9f\" xy=\"%4.9f\" xz=\"%4.9f\" yy=\"%4.9f\" yz=\"%4.9f\" zz=\"%4.9f\" xa=\"%4.9f\" xb=\"%4.9f\" xc=\"%4.9f\" ya=\"%4.9f\" yb=\"%4.9f\" yc=\"%4.9f\" za=\"%4.9f\" zb=\"%4.9f\" zc=\"%4.9f\" aa=\"%4.9f\" ab=\"%4.9f\" ac=\"%4.9f\" bb=\"%4.9f\" bc=\"%4.9f\" cc=\"%4.9f\" />\n" % (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0))
            elif input_file[iline].find('setposition') != -1:
                #if not (input_file[iline-1].find("wheel=\"0\" station=\"4") !=-1): APE = 0. #or input_file[iline-1].find("wheel=\"0\" station=\"2") !=-1): APE = 0.
                #print input_file[iline-1], " ", APE
                if dof == "xxx": output_file.write("    <setposition relativeto=\"ideal\" x=\"%4.9f\" y=\"%4.9f\" z=\"%4.9f\" phix=\"%4.9f\" phiy=\"%4.9f\" phiz=\"%4.9f\" />\n" % (0,0,0,0,0,0))
            else: 
                output_file.write(input_file[iline])   
        output_file.close()

        print "Converting the xml to db..."
        Popen("cmsRun convertXMLtoSQLite_cfg.py artifScenario_Sigma"+dof+'_'+scenLabel[scenario]+".xml",shell=True).wait()
