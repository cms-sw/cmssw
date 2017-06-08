#!/usr/bin/python

import sys
from ROOT import *
import os
from subprocess import call
import os.path
import shutil
import subprocess
import codecs


Run_type = sys.argv[1]
Run_Number = [int(x) for x in sys.argv[2:]]
CMSSW_BASE = str(os.popen('echo ${CMSSW_BASE}').read().strip())


if Run_type == 'Cosmics':
    print Run_type
elif Run_type == 'MinimumBias':
    print Run_type
elif Run_type == 'StreamExpress':
    print Run_type
elif Run_type == 'StreamExpressCosmics':
    print Run_type
else: 
    print "please enter a valid run type: Cosmics | MinimumBias | StreamExpress | StreamExpressCosmics ";
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


    nnn=Run_Number[i]/100
    nnnOut = Run_Number[i]/1000
    
    if Run_type == "Cosmics":
        checkdir='/data/users/event_display/'+DataLocalDir+'/Cosmics/'+str(nnnOut)+'/'+str(Run_Number[i])+'/StreamExpressCosmics'
        if not os.path.isdir(checkdir):
            os.makedirs(checkdir)

    else:
        checkdir='/data/users/event_display/'+DataLocalDir+'/Beam/'+str(nnnOut)+'/'+str(Run_Number[i])+'/StreamExpress'
        if not os.path.isdir(checkdir):
            os.makedirs(checkdir)


    print 'Processing '+Run_type+ ' in '+DataOfflineDir+"..."


    print 'Directory to fetch the DQM file from: https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OfflineData/'+DataOfflineDir+'/'+Run_type+'/000'+str(nnn)+'xx/'
    url = 'https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OfflineData/'+DataOfflineDir+'/'+Run_type+'/000'+str(nnn)+'xx/'
    os.popen("curl -k --cert /data/users/cctrkdata/current/auth/proxy/proxy.cert --key /data/users/cctrkdata/current/auth/proxy/proxy.cert -X GET "+url+" > index.html") 
    f=codecs.open("index.html", 'r')
    index = f.readlines()
    for s in range(len(index)): 
        if str(Run_Number[i]) in index[s]:
            if str("__DQMIO.root") in index[s]:
                File_Name=str(str(index[s]).split("xx/")[1].split("'>DQM")[0])

    print 'Downloading DQM file:'+File_Name

    
    os.system('curl -k --cert /data/users/cctrkdata/current/auth/proxy/proxy.cert --key /data/users/cctrkdata/current/auth/proxy/proxy.cert -X GET https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OfflineData/'+DataOfflineDir+'/'+Run_type+'/000'+str(nnn)+'xx/'+File_Name+' > /tmp/'+File_Name)

    filepath = '/tmp/'
    

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
        
    globalTag = str(os.popen('getGTfromDQMFile.py '+ filepath+File_Name+' ' +str(Run_Number[i])+' globalTag_Step1').readline().strip())
    print globalTag

    
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
    
# rename bad module list file
    sefile = 'QualityTest_run'+str(Run_Number[i])+'.txt'
    shutil.move('QTBadModules.log',sefile)

# put color legend in the TrackerMap     
    os.system('/usr/bin/python ${CMSSW_BASE}/src/DQM/SiStripMonitorClient/scripts/LegendToQT.py QTestAlarm.png /data/users/cctrack/FinalLegendTrans.png')
    shutil.move('result.png', 'QTestAlarm.png')



    if Run_type == "Cosmics":
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
    if Run_type != "StreamExpress":
        shutil.copyfile(sefile, checkdir+'/'+sefile)
        os.system('/usr/bin/python ${CMSSW_BASE}/src/DQM/SiStripMonitorClient/scripts/findBadModT9.py -p '+sefile+' -s /'+checkdir+'/'+sefile);
      
      
## Producing the run certification by lumisection
    
    print " Creating the lumisection certification:"

    if (Run_type == "MinimumBias") or (Run_type == "StreamExpress"):
        os.system('ls_cert 0.95 0.95 '+filepath+'/'+File_Name)

## Producing the PrimaryVertex/BeamSpot quality test by LS..
    if (Run_type != "Cosmics") and ( Run_type != "StreamExpress") and (Run_type != "StreamExpressCosmics"):
        print " Creating the BeamSpot Calibration certification summary:"
        os.system('lsbs_cert '+filepath+'/'+File_Name)

## .. and harvest the bad beamspot LS with automatic emailing (if in period and if bad LS found)
    os.system('bs_bad_ls_harvester . '+str(Run_Number[i]))
    

## Producing the Module difference for ExpressStream
    if (Run_type == "StreamExpress") or  (Run_type == "StreamExpressCosmics"):
        print " Creating the Module Status Difference summary:"

        dest='Beam'
    if (Run_type == "Cosmics") or (Run_type == "StreamExpressCosmics"):
        dest="Cosmics"


## create merged list of BadComponent from (PCL, RunInfo and FED Errors) ignore for now
    os.system('cmsRun ${CMSSW_BASE}/src/DQM/SiStripMonitorClient/test/mergeBadChannel_Template_cfg.py globalTag='+globalTag+' runNumber='+str(Run_Number[i])+' dqmFile='+filepath+'/'+File_Name)
    shutil.move('MergedBadComponents.log','MergedBadComponents_run'+str(Run_Number[i])+'.txt')

#    HOST='cctrack@vocms062'
    HOST='cctrack@vocms061'
    COMMAND="mkdir -p /data/users/event_display/TkCommissioner_runs/"+DataLocalDir+"/"+dest+" 2> /dev/null" 
    ssh = subprocess.Popen(["ssh", "%s" % HOST, COMMAND],
                       shell=False,
                        stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    ssh.stdout
    os.system('scp *.root cctrack@vocms062:/data/users/event_display/TkCommissioner_runs/'+DataLocalDir+'/'+dest)
#    os.system('scp *.root cctrack@vocms061:/data/users/event_display/TkCommissioner_runs/'+DataLocalDir+'/'+dest)
    os.remove(detIdInfoFileName)



    print "countig dead pixel ROCs" 
    if (DataLocalDir=="Data2016") or (DataLocalDir=="Data2015") or (DataLocalDir=="Data2013") or (DataLocalDir=="Data2016"):
        os.system('python ${CMSSW_BASE}/src/DQM/SiStripMonitorClient/scripts/DeadROCCounter.py '+filepath+'/'+File_Name)
    else: 
        os.system('python ${CMSSW_BASE}/src/DQM/SiStripMonitorClient/scripts/DeadROCCounter_Phase1.py '+filepath+'/'+File_Name)

    Command2='mkdir -p /data/users/event_display/'+DataLocalDir+'/'+dest+'/'+str(nnnOut)+'/'+str(Run_Number[i])+'/'+Run_type+' 2> /dev/null'

    ssh2 = subprocess.Popen(["ssh", "%s" % HOST, Command2],
                       shell=False,
                        stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    ssh2.stdout

    shutil.move('PixZeroOccROCs_run'+str(Run_Number[i])+'.txt',workPath+'/PixZeroOccROCs_run'+str(Run_Number[i])+'.txt')

#    os.system('scp -r * cctrack@vocms062:/data/users/event_display/'+DataLocalDir+'/'+dest+'/'+str(nnnOut)+'/'+str(Run_Number[i])+'/'+Run_type)
    os.system('scp -r * cctrack@vocms061:/data/users/event_display/'+DataLocalDir+'/'+dest+'/'+str(nnnOut)+'/'+str(Run_Number[i])+'/'+Run_type)

    os.chdir(workPath)
    shutil.rmtree(str(Run_Number[i]))
    os.remove('index.html')

    # produce pixel phase1 TH2Poly maps
    os.chdir(CMSSW_BASE+'/src/DQM/SiStripMonitorClient/scripts/PhaseIMaps/')
    os.system('python ${CMSSW_BASE}/src/DQM/SiStripMonitorClient/scripts/PhaseIMaps/TH2PolyOfflineMaps.py ' + filepath+'/'+File_Name+'3000 2000 limits.dat')
    shutil.move(workPath+'/PixZeroOccROCs_run'+str(Run_Number[i])+'.txt', 'OUT/PixZeroOccROCs_run'+str(Run_Number[i])+'.txt')
#    os.system('scp -r OUT/* cctrack@vocms062:/data/users/event_display/'+DataLocalDir+'/'+dest+'/'+str(nnnOut)+'/'+str(Run_Number[i])+'/'+Run_type)
    os.system('scp -r OUT/* cctrack@vocms061:/data/users/event_display/'+DataLocalDir+'/'+dest+'/'+str(nnnOut)+'/'+str(Run_Number[i])+'/'+Run_type)
    shutil.rmtree('OUT')


    # produce pixel phase1 tree for Offline TkCommissioner
    pixelTreeFileName = 'PixelPhase1Tree_Run'+str(Run_Number[i])+'_'+Run_type+'.root'
    os.system('python ${CMSSW_BASE}/src/DQM/SiStripMonitorClient/scripts/PhaseIMaps/PhaseITreeProducer.py ' + filepath+'/'+File_Name + ' DATA/detids.dat ' + pixelTreeFileName)

#    os.system('scp ' + pixelTreeFileName + ' cctrack@vocms062:/data/users/event_display/TkCommissioner_runs/'+DataLocalDir+'/'+dest)
    os.system('scp ' + pixelTreeFileName + ' cctrack@vocms061:/data/users/event_display/TkCommissioner_runs/'+DataLocalDir+'/'+dest)
 
    os.remove(pixelTreeFileName)

    os.chdir(workPath)

