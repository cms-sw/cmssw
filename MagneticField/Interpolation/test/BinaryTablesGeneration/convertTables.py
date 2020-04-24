#!/bin/env python

import os

# Path of the input tables
INPUTPATH = '/afs/cern.ch/cms/OO/mag_field/versions_new/version_18l_160812_3_8t_v9_small_fin'
#INPUTPATH = '/afs/cern.ch/cms/OO/mag_field/versions_new/version_18l_160812_3t_v9_large_fin'
#INPUTPATH = '/afs/cern.ch/cms/OO/mag_field/versions_new/version_18l_160812_3_5t_v9_large_fin'
#INPUTPATH = '/afs/cern.ch/cms/OO/mag_field/versions_new/version_18l_160812_3_8t_v9_large_fin'


f = open('tableList.txt', 'w') #will write a list of tables here, for further validation scripts
for part in range (1, 3) : 
    for sector in range (1, 13) :
        subdir = 's'+str(sector).zfill(2)
        os.system('mkdir -p '+subdir)
        for volume in range (1, 465) :
            volNo=str(volume+1000*part)
            type='rpz' # for Tubs, Cone, TruncTubs
            if (volume>=138 and volume <= 402) :
                type = 'xyz' #for Box, Trapezoid         
            fullpath = INPUTPATH+'/'+subdir+'_'+str(part)+'/v-'+type+'-'+volNo+'.table'
            f.write(fullpath+'\n')
            status = os.system('${CMSSW_BASE}/test/${SCRAM_ARCH}/prepareFieldTable '+fullpath +' '+ subdir+'/grid.'+volNo+'.bin '+ str(sector))
            if status != 0 :
                print 'ERROR table not processed:', fullpath
        
f.close()
