#!/usr/bin/env python

import os
import sys

runnumber = sys.argv[2]
dataset = sys.argv[1]

basedir = '/pnfs/cms/WAX/11/store/data/'+dataset+'/A/000'
runnumber2 = runnumber[-3:]
runnumber1 = runnumber[:-3]
#print runnumber1, runnumber2

if (len(runnumber1) == 2):
    runnumber1 = '0'+runnumber1

rawfilelist =  os.listdir(basedir+'/'+runnumber1+'/'+runnumber2)
filelist = []
while rawfilelist:
    if (rawfilelist[0][-4:] == '.dat'):
        filelist.append(basedir+'/'+runnumber1+'/'+runnumber2+'/'+rawfilelist[0])   
    rawfilelist = rawfilelist[1:]
filelist.sort()
    
outputfile = open('inputfiles.cff', 'w')
outputfile.write('source = NewEventStreamFileReader{\n')
outputfile.write('untracked vstring fileNames = {\n')


while filelist:
    print filelist[0]
    if len(filelist) > 1:
        outputfile.write('\"dcache:'+filelist[0]+'\",\n')
    if len(filelist) == 1:
        outputfile.write('\"dcache:'+filelist[0]+'\"\n')
    filelist = filelist[1:]
    
outputfile.write('}\n')
outputfile.write('int32 max_event_size = 7000000\n')
outputfile.write('int32 max_queue_depth = 5\n')
outputfile.write('}\n')
outputfile.close()

os.system("cmsRun DQM/L1TMonitor/data/L1TDQM_GlobalNov07_run.cfg")


