#!/usr/bin/python
import os
import re
import subprocess

##Setting variables:
webDir = '/data/users/event_display/HDQM/Current/'
webDir = '/data/users/event_display/HDQM/dev2/'
#Epochs = ['Run2012A','Run2012B','Run2012']
Epochs = ['Run2012B']
Recos  = ['Prompt']
PDs    = ['MinimumBias']      ##other examples: 'SingleMu','DoubleMu'
jsonFile  = subprocess.Popen("ls -1tr /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions12/8TeV/Prompt/Cert_*_8TeV_PromptReco_Collisions12_JSON.txt", shell=True, stdout=subprocess.PIPE).stdout.readlines()[-1][:-1]
lastwkfile= '/afs/cern.ch/user/c/cctrack/LastWeekRuns_SLlist.txt'

##Internally set vars
pwDir  = subprocess.Popen("pwd", shell=True, stdout=subprocess.PIPE).stdout.readline()[:-1]+'/'
YEAR=20100
if YEAR==2010:
    jsonFile = subprocess.Popen("ls -1tr /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions10/7TeV/Reprocessing/Cert_136033-149442_7TeV_Apr21ReReco_Collisions10_JSON.txt", shell=True, stdout=subprocess.PIPE).stdout.readlines()[-1][:-1]
    Epochs = ['Run2010','Run2010A','Run2010B']
    Epochs = ['Run2010B']
    Recos  = ['Dec22ReReco']
if YEAR==2011:
    jsonFile = subprocess.Popen("ls -1tr /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions11/7TeV/Reprocessing/Cert_160404-180252_7TeV_ReRecoNov08_Collisions11_JSON_v2.txt", shell=True, stdout=subprocess.PIPE).stdout.readlines()[-1][:-1]
    Epochs = ['Run2011','Run2011A','Run2011B']
    Recos  = ['08Nov2011-v','19Nov2011-v']
    Recos  = ['Nov2011-v']

#Cleaning up areas of last night's plots
#Tomislav: don't do that here since yu just want to rm those dir which you produce
#rmCmd='rm -rf  '+webDir+'fig/'
#subprocess.Popen(rmCmd, shell=True).wait()


subprocess.Popen("rm -rf fig/*", shell=True).wait()
#copy over strip mode map
subprocess.Popen("cp /data/users/cctrack/DQMdata/StripMode/StripReadoutMode4Cosmics.txt ./", shell=True).wait()
##Internally set vars
pwDir  = subprocess.Popen("pwd", shell=True, stdout=subprocess.PIPE).stdout.readline()[:-1]+'/'

addplots=''
Plots=[' -C cfg/trendPlotsTracking.ini ',' -C cfg/trendPlotsPixel_General.ini',' -C cfg/trendPlotsStrip_APVShots.ini -C cfg/trendPlotsStrip_General.ini -C cfg/trendPlotsStrip_TEC.ini -C cfg/trendPlotsStrip_TIB.ini -C cfg/trendPlotsStrip_TID.ini -C cfg/trendPlotsStrip_TOB.ini -C cfg/trendPlotsStrip_Clusters.ini', ' -C cfg/trendPlotsRECOErrors.ini']
#Plots=[' -C cfg/trendPlotsTracking.ini ',' -C cfg/trendPlotsPixel_General.ini',' -C cfg/trendPlotsStrip_APVShots.ini']
title=['Tracking_plots','Pixel_plots','Strip_plots','RECOError_plots']
indexOut=['index_tracker.htm','index_pixel.htm','index_strip.htm','index_reco.htm']
cosType='ALL'
##Loop hDQM with xxx PDs
for epoch in Epochs:
    epochOut = epoch
    for reco in Recos:
        for pd in PDs:
          for i in range(0,len(Plots)):
            addplots=Plots[i]
            #if 'Cosmics' in pd:
   	    if (('Cosmics' in pd) or ('StreamExpressCosmics' in pd)):
		for icos in range(0,2):
                  ##Running both PEAK + DECO mode...
		  if int(icos)==int(0): cosType='PEAK'
		  if int(icos)==int(1): cosType='DECO'
        	  if 'StreamExpressCosmics' in pd: reco='Express'
		  run_hDQM_cmd = './trendPlots.py  -C cfg/trendPlotsDQM.ini' + addplots +  ' --epoch '+epoch+' --dataset '+pd+' --reco '+reco +" -s "+cosType
		  if 'LastWeek' in epoch:
                        epoch='Run2012C'
                        reco='Prompt'
                        epochOut='LastWeek'
                        run_hDQM_cmd = './trendPlots.py  -C cfg/trendPlotsDQM.ini' + addplots +  ' --epoch '+epoch+' --dataset '+pd+' --reco '+reco +' -s '+cosType + ' -L '+lastwkfile
		  print 'RUN:',run_hDQM_cmd, icos
                  ##Define local output directory
		  figDir=pwDir+'fig/'+reco+'/'+epoch+'/'+pd+'/'+cosType+'/'
                  ##Run trendplots
		  subprocess.Popen(run_hDQM_cmd, shell=True).wait()
                  ##Define web directory
		  outPath=webDir+'fig/'+reco+'/'+epoch+'/'+pd+'/'+cosType+'/'+title[i]+'/'
		  if 'LastWeek' in epochOut:
                        outPath=webDir+'fig/'+reco+'/'+epochOut+'/'+pd+'/'+cosType+'/'+title[i]+'/'
                  rmCmd='rm -r  '+ outPath
		  subprocess.Popen(rmCmd, shell=True)
                  ##Make web directory if it doesn't exist
                  if not os.path.exists(outPath): 
			os.makedirs(outPath)
		  else:
                  	subprocess.Popen("rm -r "+outPath, shell=True).wait()
			os.makedirs(outPath)
                  ##Copy all files to output directory
		  mvCmd="cp "+figDir+"*.png  "+outPath
		  print "MOVE:", mvCmd
		  subprocess.Popen(mvCmd, shell=True).wait()
                  ##Merge rootfiles for associated plots into one directory
		  inoneCmd="python test/get_all_plots_in_single_file.py "+figDir+ "  "+outPath+title[i]+"_all_plots.root"
		  subprocess.Popen(inoneCmd, shell=True).wait()
                  ##Finally, make ratios if RECO is in title[i]...
                  if 'RECO' in title[i]:
                      ##remove all current pngs in web dir...
                      subprocess.Popen("rm -f "+outPath+"*.png", shell=True).wait()
                      ##make ratio plots for reco errors
                      subprocess.Popen("python test/makeRecoErrRatios.py "+outPath+" -b", shell=True).wait()
                  subprocess.Popen("rm "+figDir+"* -f", shell=True).wait()
		  if 'LastWeek' in epochOut:
                        epoch='LastWeek'
            else :
                run_hDQM_cmd = './trendPlots.py  -C cfg/trendPlotsDQM.ini' + addplots +  ' --epoch '+epoch+' --dataset '+pd+' --reco '+reco +" -J "+jsonFile
                ##Define web directory
		outPath=webDir+'fig/'+reco+'/'+epoch+'/'+pd+'/'+title[i]+'/'
    #        run_hDQM_cmd+='  -r \"run > 190644 and run < 191047\"'
                if 'LastWeek' in epoch:
                        epoch='Run2012C'
                        reco='Prompt'
                        epochOut='LastWeek'
                        outPath=webDir+'fig/'+reco+'/'+epochOut+'/'+pd+'/'+title[i]+'/'
                        print 'outPath22:',outPath
                        run_hDQM_cmd = './trendPlots.py  -C cfg/trendPlotsDQM.ini' + addplots +  ' --epoch '+epoch+' --dataset '+pd+' --reco '+reco +' -L '+lastwkfile
            	print "Running ",run_hDQM_cmd
	        print "pwDir",pwDir
                ##Define local output directory
                figDir=pwDir+'fig/'+reco+'/'+epoch+'/'+pd+'/' 
                print "figDir",figDir
                ##run trendplots
                subprocess.Popen(run_hDQM_cmd, shell=True).wait()
                print 'outPath',outPath
                ##Make web directory if it doesn't exist
                if not os.path.exists(outPath): 
			os.makedirs(outPath) 
		else:
			subprocess.Popen("rm -r "+outPath, shell=True).wait()
                        os.makedirs(outPath)
                mvCmd="cp "+figDir+"*.png  "+outPath
                print 'mvCmd',mvCmd
                subprocess.Popen(mvCmd, shell=True).wait()
                ##Merge rootfiles for associated plots into one directory
                inoneCmd="python test/get_all_plots_in_single_file.py "+figDir+ "  "+outPath+title[i]+"_all_plots.root"
                print 'inoneCmd',inoneCmd
                subprocess.Popen(inoneCmd, shell=True).wait()
                if 'RECO' in title[i]:
                    ##remove all current pngs in web dir...
                    subprocess.Popen("rm -f "+outPath+"*.png", shell=True).wait()
                    ##make ratio plots for reco errors
                    subprocess.Popen("python test/makeRecoErrRatios.py "+outPath+ " -b", shell=True).wait()
                ##Next, make indices for all of the output plots
                subprocess.Popen("rm "+figDir+"* -f", shell=True).wait()
                perlCmd = "perl "+pwDir+"test/diowroot2.pl -c 2 -t "+title[i] +" -D "+outPath +" -o "+webDir +indexOut[i]
                print "perlCmd:", perlCmd
                #subprocess.Popen(perlCmd, shell=True).wait()
		if (('MinimumBias' in pd) and ('2012B' in epoch)):
                    subprocess.Popen(perlCmd, shell=True).wait()
		if 'LastWeek' in epochOut:
                        epoch='LastWeek'

                ##Finally, cp generic hDQM index for basic plots
                ####Need to get that file from Tomislav
