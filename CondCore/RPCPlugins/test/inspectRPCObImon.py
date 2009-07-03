#!/usr/bin/env python

import os,sys, DLFCN, time, datetime
import matplotlib
import numpy
import pylab
import math
import cherrypy

from pylab import figure, show
matplotlib.use('Agg')

sys.setdlopenflags(DLFCN.RTLD_GLOBAL+DLFCN.RTLD_LAZY)

from pluginCondDBPyInterface import *

a = FWIncantation()
rdbms = RDBMS()
dbName =  "sqlite_file:dati.db"
rdbms = RDBMS("/afs/cern.ch/cms/DB/conddb/")
dbName =  "oracle://cms_orcoff_prep/CMS_COND_30X_RPC"
tagPVSS = 'test1'
        
from CondCore.Utilities import iovInspector as inspect

db = rdbms.getDB(dbName)
tags = db.allTags()

##----------------------- Create assosiacion map for detectors -----------------
##tagPVSS = 'PVSS_v3'

try:
    iov = inspect.Iov(db,tagPVSS)
    iovlist=iov.list()
    print iovlist
    detMapName = {}

    for p in iovlist:
        payload=inspect.PayLoad(db,p[0])
        info = payload.summary().split(" ")

        i = 0
        for e in info:
            
            try:
                if int(info[i+1]) == 0:
                    detName = "W"+str(info[i+2])+"_S"+str(info[i+3])+"_RB"+str(info[i+4])+"_Layer"+str(info[i+5])+"_SubS"+str(info[i+6])
                else:
                    detName = "D"+str(info[i+2])+"_S"+str(info[i+3])+"_RB"+str(info[i+4])+"_Layer"+str(info[i+5])+"_SubS"+str(info[i+6])

                detMapName[info[i]] = detName
                i += 7
            except:
                pass

    for (k,v) in detMapName.items():
        print k,v


except Exception, er :
    print er

##--------------------- Current reading -----------------------------------------

##tag = 'Imon_v3'

####tag = 'Test_70195'
##timeStart = time.mktime((2008, 11, 9, 4, 10, 0,0,0,0))
##timeEnd = time.mktime((2008, 11, 10, 5, 10, 0,0,0,0))

####print timeStart,timeEnd, time.ctime(timeStart),time.ctime(timeEnd)

##try:

##    iov1 = inspect.Iov(db,tag)
##    iovlist1=iov1.list()
##    detMap = {}
    
##    for p1 in iovlist1:
##        payload1=inspect.PayLoad(db,p1[0])
##        info = payload1.summary().split(" ")

##        i = 0
##        for e in info:
##            if i+2 < len(info):
##                timein = int(info[i+2])+(p1[1] >> 32)
##                if timein < int(timeEnd):
##                    if detMap.has_key(info[i]):
##                        detMap[info[i]][0].append(timein)
##                        detMap[info[i]][1].append(float(info[i+1]))
##                    else:
##                        detMap[info[i]] = [[timein],[float(info[i+1])]]

####                    if detMap.has_key((detMapName[info[i]])):
####                        print "Exist the key!"
####                        print "Current vaues: ", detMap[(detMapName[info[i]])][0], detMap[(detMapName[info[i]])][1]
####                        print "value is: ", timein, float(info[i+1])
####                        print "The det name is: ", (detMapName[info[i]])
####                        
####                        detMap[(detMapName[info[i]])][0].append(timein)
####                        detMap[(detMapName[info[i]])][1].append(float(info[i+1]))
####                    else:
####                        print "Not Exist the key!"
####                        print "Value is: ", timein, float(info[i+1])
####                        print "Det id: ", info[i]
####                        print "The det name is: ", (detMapName[info[i]])
####                        
####                        detMap[(detMapName[info[i]])] = [[timein],[float(info[i+1])]]
                    
##                i += 3

##    for (k,v) in  detMap.items():

##        f=pylab.figure()
##        ax = f.add_subplot(111)
##        ax.set_xlabel('Time (s)')
##        ax.set_ylabel('Current (uA)')
##        ax.set_title(str(k))
##        ax.plot(v[0],v[1],'rs-',linewidth=2)
##        f.canvas.draw()
##        f.savefig('/tmp/trentad/'+str(k),format='png')

##        average = pylab.polyfit(v[0],v[1],0)
##        sigma = 0
##        highCurrent = False
        
##        for c in v[1]:
##            sigma += math.pow((average[0]-c), 2)
##            if math.fabs(average[0]-c) > 1: highCurrent = True
##        sigma = math.sqrt(sigma/len(v[1]))

##        if  highCurrent:
##            print "Num points: ",len(v[0]),"   Fit coeff: ",pylab.polyfit(v[0],v[1],0), "   Sigma: ",sigma, "   HighCurrrent: ", highCurrent
        

##except Exception, er :
##    print er


