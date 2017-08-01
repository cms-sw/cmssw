#!/usr/bin/python

import sys
from ROOT import *
import os
from subprocess import call
import os.path
import shutil
import subprocess
import codecs
import re
import errno



Run_type = sys.argv[1]
Run_Number = [int(x) for x in sys.argv[2:]]
CMSSW_BASE = str(os.popen('echo ${CMSSW_BASE}').read().strip())

###########Check if user enter the right run type######################
if Run_type == 'Cosmics' or Run_type == 'StreamExpress' or Run_type == 'StreamExpressCosmics' or 'ZeroBias' or re.match('ZeroBias([0-9]+?)',Run_type):

    print Run_type
else: 
    print "please enter a valid run type: Cosmics | ZeroBias | StreamExpress | StreamExpressCosmics ";
    sys.exit(0)

#########Checking Data taking period##########
for i in range(len(Run_Number)):

    if Run_Number[i] > 294644:
        DataLocalDir='Data2017'
        DataOfflineDir='Run2017'

    elif Run_Number[i] > 290123:
        DataLocalDir='Data2017'
        DataOfflineDir='Commissioning2017'


    elif Run_Number[i] > 284500:
        DataLocalDir='Data2016';
        DataOfflineDir='PARun2016';

##2016 data taking period run > 271024
    elif Run_Number[i] > 271024:
        DataLocalDir='Data2016';
        DataOfflineDir='Run2016';

#2016 - Commissioning period                                                                                                                               
    elif Run_Number[i] > 264200:
        DataLocalDir='Data2016';
        DataOfflineDir='Commissioning2016';

#Run2015A
    elif Run_Number[i] > 246907:
        DataLocalDir='Data2015';
        DataOfflineDir='Run2015'
    
#2015 Commissioning period (since January)
    elif Run_Number[i] > 232881:
        DataLocalDir='Data2015';
        DataOfflineDir='Commissioning2015';

#2013 pp run (2.76 GeV)
    elif Run_Number[i] > 211658:
        DataLocalDir='Data2013';
        DataOfflineDir='Run2013';

#2013 HI run
    elif Run_Number[i] > 209634:
        DataLocalDir='Data2013';
        DataOfflineDir='HIRun2013';
   
    elif Run_Number[i] > 190450:
        DataLocalDir='Data2012';
        DataOfflineDir='Run2012';

    else:
        print "Please enter vaild run numbers"
        sys.exit(0)

#################Downloading DQM file############################
    nnn=Run_Number[i]/100
    nnnOut = Run_Number[i]/1000
    nnnOnline = Run_Number[i]/10000

    print 'Processing '+Run_type+ ' in '+DataOfflineDir+"..."

    File_Name = ''

    print 'Directory to fetch the DQM file from: https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OfflineData/'+DataOfflineDir+'/'+Run_type+'/000'+str(nnn)+'xx/'
    url = 'https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OfflineData/'+DataOfflineDir+'/'+Run_type+'/000'+str(nnn)+'xx/'
    os.popen("curl -k --cert /data/users/cctrkdata/current/auth/proxy/proxy.cert --key /data/users/cctrkdata/current/auth/proxy/proxy.cert -X GET "+url+" > index.html") 
    f=codecs.open("index.html", 'r')
    index = f.readlines()
    if any(str(Run_Number[i]) in s for s in index):
        for s in index:
            if (str(Run_Number[i]) in s) and ("__DQMIO.root" in s):
                File_Name = str(str(s).split("xx/")[1].split("'>DQM")[0])
    else:
        print 'No DQM file available. Please check the Offline server'
        sys.exit(0)

    print 'Downloading DQM file:'+File_Name

    
    os.system('curl -k --cert /data/users/cctrkdata/current/auth/proxy/proxy.cert --key /data/users/cctrkdata/current/auth/proxy/proxy.cert -X GET https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OfflineData/'+DataOfflineDir+'/'+Run_type+'/000'+str(nnn)+'xx/'+File_Name+' > /tmp/'+File_Name)

    filepath = '/tmp/'



##################online file########    
    url1 = 'https://cmsweb.cern.ch/dqm/online/data/browse/Original/000'+str(nnnOnline)+'xxxx/000'+str(nnn)+'xx/'
    os.popen("curl -k --cert /data/users/cctrkdata/current/auth/proxy/proxy.cert --key /data/users/cctrkdata/current/auth/proxy/proxy.cert -X GET "+url1+" > index_online.html")
    
    url2 = 'https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OnlineData/original/000'+str(nnnOnline)+'xxxx/000'+str(nnn)+'xx/'
    os.popen("curl -k --cert /data/users/cctrkdata/current/auth/proxy/proxy.cert --key /data/users/cctrkdata/current/auth/proxy/proxy.cert -X GET "+url2+" > index_online_backup.html")
    f_online_backup=codecs.open("index_online_backup.html", 'r')
    index_online_backup = f_online_backup.readlines()

    f_online=codecs.open("index_online.html", 'r')
    index_online = f_online.readlines()
    if any(str(Run_Number[i]) in x for x in index_online):
        for x in index_online:
            if (str(Run_Number[i]) in x) and ("_PixelPhase1_" in x):
                File_Name_online=str(str(x).split(".root'>")[1].split("</a></td><td>")[0])
                deadRocMap = True
    else:
        print "Can't find any file in offline server, trying the online server"
        if any(str(Run_Number[i]) in y for y in index_online_backup):
            for y in index_online:
                if (str(Run_Number[i]) in y) and ("_PixelPhase1_" in y):
                    File_Name_online=str(str(y).split(".root'>")[1].split("</a></td><td>")[0])
                    deadRocMap = True
        else:
            print 'No Online DQM file available. Skip dead roc map'
            deadRocMap = False

    print 'Downloading DQM file:'+File_Name_online


    os.system('curl -k --cert /data/users/cctrkdata/current/auth/proxy/proxy.cert --key /data/users/cctrkdata/current/auth/proxy/proxy.cert -X GET https://cmsweb.cern.ch/dqm/online/data/browse/Original/000'+str(nnnOnline)+'xxxx/000'+str(nnn)+'xx/'+File_Name_online+' > /tmp/'+File_Name_online)

    os.remove('index_online.html')
    os.remove('index_online_backup.html')




################Check if run is complete##############

    print "get the run status from DQMFile"


    check_command = 'check_runcomplete '+filepath+File_Name
    Check_output = subprocess.call(check_command, shell=True)


    if Check_output == 0:
        print 'Using DQM file: '+File_Name
    else:
        print '*****************Warning: DQM file is not ready************************';
        input_var = raw_input("DQM file is incompleted, do you want to continue? (y/n): ")
        if (input_var == 'y') or (input_var == 'Y'):
            print 'Using DQM file: '+File_Name
        else:
            sys.exit(0)

    
###################Start making TkMaps################

    checkfolder = os.path.exists(str(Run_Number[i]))
    if checkfolder == True:
        shutil.rmtree(str(Run_Number[i]))
        os.makedirs(str(Run_Number[i])+'/'+Run_type)
    else:
        os.makedirs(str(Run_Number[i])+'/'+Run_type)
        
#######**Temporary solution**##############
    ####After switch production to 9_3_X release, please comment out this section ##########
    globalTag_v0 = os.popen('getGTfromDQMFile.py '+ filepath+File_Name+' ' +str(Run_Number[i])+' globalTag_Step1').readline().strip()
    print "Global Tag: " + globalTag_v0
    
    globalTag = raw_input("Please enter the GlobalTag that shows above: ")
    ########################################################################################

    ######and uncomment this two lines##################
#    globalTag = os.popen('getGTfromDQMFile.py '+ filepath+File_Name+' ' +str(Run_Number[i])+' globalTag_Step1').readline().strip()
#    print "Global Tag: " + globalTag
    ####################################################

    if globalTag == "":
        print " No GlobalTag found: trying from DAS.... ";
        globalTag = str(os.popen('getGTscript.sh '+filepath+ File_Name+' ' +str(Run_Number[i])));
        if globalTag == "":
            print " No GlobalTag found for run: "+str(Run_Number[i]);
    
    print " Creating the TrackerMap.... "

    detIdInfoFileName = 'TkDetIdInfo_Run'+str(Run_Number[i])+'_'+Run_type+'.root'
    workPath = os.popen('pwd').readline().strip()

    os.chdir(str(Run_Number[i])+'/'+Run_type)
   

    os.system('cmsRun ${CMSSW_BASE}/src/DQM/SiStripMonitorClient/test/SiStripDQM_OfflineTkMap_Template_cfg_DB.py globalTag='+globalTag+' runNumber='+str(Run_Number[i])+' dqmFile='+filepath+'/'+File_Name+' detIdInfoFile='+detIdInfoFileName)
    os.system('rm -f *svg')
####################### rename bad module list file######################
    sefile = 'QualityTest_run'+str(Run_Number[i])+'.txt'
    shutil.move('QTBadModules.log',sefile)

################### put color legend in the TrackerMap###################

# PLEASE UNCOMMENT THE LINES BELOW TO GET THE LEGEND ON THE QT TkMAP (it will work only on vocms061)    
#    os.system('/usr/bin/python ${CMSSW_BASE}/src/DQM/SiStripMonitorClient/scripts/LegendToQT.py QTestAlarm.png /data/users/cctrack/FinalLegendTrans.png')
#    shutil.move('result.png', 'QTestAlarm.png')

####################Copying the template html file to index.html########################

    if Run_type == "Cosmics" or Run_type == "StreamExpressCosmics":
        os.system('cat ${CMSSW_BASE}/src/DQM/SiStripMonitorClient/data/index_template_TKMap_cosmics.html | sed -e "s@RunNumber@'+str(Run_Number[i])+'@g" > index.html')
    else:
        os.system('cat ${CMSSW_BASE}/src/DQM/SiStripMonitorClient/data/index_template_TKMap.html | sed -e "s@RunNumber@'+str(Run_Number[i])+'@g" > index.html')

    shutil.copyfile(CMSSW_BASE+'/src/DQM/SiStripMonitorClient/data/fedmap.html','fedmap.html')
    shutil.copyfile(CMSSW_BASE+'/src/DQM/SiStripMonitorClient/data/psumap.html','psumap.html')

    print " Check TrackerMap on "+str(Run_Number[i])+'/'+Run_type+" folder"
    
    output =[]
    output.append(os.popen("/bin/ls ").readline().strip())
    print output

## Producing the list of bad modules
    print " Creating the list of bad modules "
    
    os.system('listbadmodule '+filepath+'/'+File_Name+' PCLBadComponents.log')

 ##   if Run_type != "StreamExpress":
 ##       shutil.copyfile(sefile, checkdir+'/'+sefile)
 ##       os.system('/usr/bin/python ${CMSSW_BASE}/src/DQM/SiStripMonitorClient/scripts/findBadModT9.py -p '+sefile+' -s /'+checkdir+'/'+sefile);
      
      
## Producing the run certification by lumisection
    
#    print " Creating the lumisection certification:"

#    if (Run_type.startswith("ZeroBias")) or (Run_type == "StreamExpress"):
#        os.system('ls_cert 0.95 0.95 '+filepath+'/'+File_Name)

## Producing the PrimaryVertex/BeamSpot quality test by LS..
#    if (Run_type != "Cosmics") and ( Run_type != "StreamExpress") and (Run_type != "StreamExpressCosmics"):
#        print " Creating the BeamSpot Calibration certification summary:"
#        os.system('lsbs_cert '+filepath+'/'+File_Name)

## .. and harvest the bad beamspot LS with automatic emailing (if in period and if bad LS found)
    os.system('bs_bad_ls_harvester . '+str(Run_Number[i]))
    

## Producing the Module difference for ExpressStream

    dest='Beam'

    if (Run_type == "Cosmics") or (Run_type == "StreamExpressCosmics"):
        dest="Cosmics"


## create merged list of BadComponent from (PCL, RunInfo and FED Errors) ignore for now
    os.system('cmsRun ${CMSSW_BASE}/src/DQM/SiStripMonitorClient/test/mergeBadChannel_Template_cfg.py globalTag='+globalTag+' runNumber='+str(Run_Number[i])+' dqmFile='+filepath+'/'+File_Name)
    shutil.move('MergedBadComponents.log','MergedBadComponents_run'+str(Run_Number[i])+'.txt')

    os.system("mkdir -p /data/users/event_display/TkCommissioner_runs/"+DataLocalDir+"/"+dest+" 2> /dev/null")


    shutil.copyfile(detIdInfoFileName,'/data/users/event_display/TkCommissioner_runs/'+DataLocalDir+'/'+dest+'/'+detIdInfoFileName)

    os.remove(detIdInfoFileName)
    os.remove('MergedBadComponentsTkMap_Canvas.root')
    os.remove('MergedBadComponentsTkMap.root')
##############counting dead pixel#######################
    print "countig dead pixel ROCs" 
    if (Run_Number[i] < 290124) :

        os.system('${CMSSW_BASE}/src/DQM/SiStripMonitorClient/scripts/DeadROCCounter.py '+filepath+'/'+File_Name)
    else: 
        os.system('${CMSSW_BASE}/src/DQM/SiStripMonitorClient/scripts/DeadROCCounter_Phase1.py '+filepath+'/'+File_Name)

    os.system('mkdir -p /data/users/event_display/'+DataLocalDir+'/'+dest+'/'+str(nnnOut)+'/'+str(Run_Number[i])+'/'+Run_type+' 2> /dev/null')
    
    shutil.move('PixZeroOccROCs_run'+str(Run_Number[i])+'.txt',workPath+'/PixZeroOccROCs_run'+str(Run_Number[i])+'.txt')




######Counting Dead ROCs during the run#########################
    
    if deadRocMap == True:

        os.system('${CMSSW_BASE}/src/DQM/SiStripMonitorClient/scripts/DeadROC_duringRun.py '+filepath+File_Name_online+' '+filepath+File_Name)
        os.system('${CMSSW_BASE}/src/DQM/SiStripMonitorClient/scripts/change_name.py')
        os.system('${CMSSW_BASE}/src/DQM/SiStripMonitorClient/scripts/PixelMapPlotter.py MaskedROC_sum.txt -c')
        os.remove('MaskedROC_sum.txt')
    else:
        print 'No Online DQM file available, Dead ROC maps will not be produced'


########Counting Inefficient DC during the run #################


    if deadRocMap == True:
      
        os.system('${CMSSW_BASE}/src/DQM/SiStripMonitorClient/scripts/InefficientDoubleROC.py '+filepath+File_Name_online)
      
    else:
        print 'No Online DQM file available, inefficient DC list  will not be produced'





###################copy ouput files###################
    strip_files = os.listdir('.')
    for file_name in strip_files:
        full_stripfile_name = os.path.join('.', file_name)
        if (os.path.isfile(full_stripfile_name)):
            shutil.copy(full_stripfile_name, '/data/users/event_display/'+DataLocalDir+'/'+dest+'/'+str(nnnOut)+'/'+str(Run_Number[i])+'/'+Run_type)




#########################Start making pixel maps#####################
    os.chdir(workPath)
    shutil.rmtree(str(Run_Number[i]))
    os.remove('index.html')

    # produce pixel phase1 TH2Poly maps
#    os.chdir(CMSSW_BASE+'/src/DQM/SiStripMonitorClient/scripts/PhaseIMaps/')
    os.system('${CMSSW_BASE}/src/DQM/SiStripMonitorClient/scripts/TH2PolyOfflineMaps.py ' + filepath+'/'+File_Name+' 3000 2000')
    shutil.move(workPath+'/PixZeroOccROCs_run'+str(Run_Number[i])+'.txt', 'OUT/PixZeroOccROCs_run'+str(Run_Number[i])+'.txt')


###################copy ouput files##########
    pixel_files = os.listdir('./OUT')
    for file_name in pixel_files:
        full_pixelfile_name = os.path.join('./OUT/', file_name)
        if (os.path.isfile(full_pixelfile_name)):
            shutil.copy(full_pixelfile_name, '/data/users/event_display/'+DataLocalDir+'/'+dest+'/'+str(nnnOut)+'/'+str(Run_Number[i])+'/'+Run_type)


    shutil.rmtree('OUT')


    # produce pixel phase1 tree for Offline TkCommissioner
    pixelTreeFileName = 'PixelPhase1Tree_Run'+str(Run_Number[i])+'_'+Run_type+'.root'
    os.system('${CMSSW_BASE}/src/DQM/SiStripMonitorClient/scripts/PhaseITreeProducer.py ' + filepath+'/'+File_Name + ' ' + pixelTreeFileName)

    shutil.copyfile(pixelTreeFileName,'/data/users/event_display/TkCommissioner_runs/'+DataLocalDir+'/'+dest+'/'+pixelTreeFileName)
    os.remove(pixelTreeFileName)
    os.remove(filepath+File_Name)
    os.remove(filepath+File_Name_online)
    os.chdir(workPath)

