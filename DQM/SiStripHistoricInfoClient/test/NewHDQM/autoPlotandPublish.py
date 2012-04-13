#!/usr/bin/python

import re
import subprocess

##Setting variables:
webDir = '/data/users/event_display/HDQM/Current/'
Epochs = ['Run2011A','Run2011B']
Recos  = ['Prompt']           ##other examples: 08Nov2011
PDs    = ['Jet','SingleMu','DoubleMu']      ##other examples: 'SingleMu','DoubleMu'

##Internally set vars
pwDir  = subprocess.Popen("pwd", shell=True, stdout=subprocess.PIPE).stdout.readline()[:-1]+'/'

subprocess.Popen("rm -rf fig/*", shell=True)

##Loop hDQM with xxx PDs
for epoch in Epochs:
    for reco in Recos:
        for pd in PDs:
            #run_hDQM_cmd = './trendPlots.py -C cfg/trendPlotsDQM.ini -C cfg/trendPlotsTracker.ini  -r "run > 161305" --tag v1 --epoch '+epoch+' --dataset '+pd+' --reco '+reco
            run_hDQM_cmd = './trendPlots.py -C cfg/trendPlotsDQM.ini -C cfg/trendPlotsTracker.ini --epoch '+epoch+' --dataset '+pd+' --reco '+reco
            print "Running ",run_hDQM_cmd
            subprocess.Popen(run_hDQM_cmd, shell=True).wait()

####Build individual indices for each PD set of plots, and parent directories
epochDirs  = subprocess.Popen("ls -d1 ./fig/*", shell=True, stdout=subprocess.PIPE).stdout.readlines()
epochIndex = ''

for edir in epochDirs:
    epochIndex += '<LI><A href="'+re.split('/',edir[:-1])[-1]+'">'+re.split('/',edir[:-1])[-1]+'<\\/a>\\n'
    recoDirs  = subprocess.Popen("ls -d1 "+edir[:-1]+"/*", shell=True, stdout=subprocess.PIPE).stdout.readlines()
    recoIndex = ''
    for rdir in recoDirs:
        recoIndex += '<LI><A href="'+re.split('/',rdir[:-1])[-1]+'">'+re.split('/',rdir[:-1])[-1]+'<\\/a>\\n'
        pdDirs     = subprocess.Popen("ls -d1 "+rdir[:-1]+"/*", shell=True, stdout=subprocess.PIPE).stdout.readlines()
        pdIndex    = ''
        for pdir in pdDirs:
            ##First setup title (PD,Epoch,Reco)
            title = re.split('/',pdir[:-1])[-1]+'_'+re.split('/',edir[:-1])[-1]+'-'+re.split('/',rdir[:-1])[-1]
            ##To run, I must be in the directory with all pngs...#
            perlCmd = "perl "+pwDir+"test/diowroot.pl -c 2 -t "+title
            print perlCmd
            dir = pwDir+pdir[:-1]
            subprocess.Popen(perlCmd, shell=True).wait()
            subprocess.Popen(perlCmd, cwd=dir, shell=True).wait()
            pdIndex += '<LI><A href="'+re.split('/',pdir[:-1])[-1]+'">'+re.split('/',pdir[:-1])[-1]+'<\\/a>\\n'
        mytitle    = "hDQM "+re.split('/',edir[:-1])[-1]+'-'+re.split('/',rdir[:-1])[-1]
        mycontents = "hDQM Trend Plots "+re.split('/',edir[:-1])[-1]+'-'+re.split('/',rdir[:-1])[-1]
        subprocess.Popen("cp ./test/Template_Index.html "+rdir[:-1]+"/index.html", shell=True).wait()
        subprocess.Popen("sed -i 's/MYTITLE/"+mycontents+"/g' "+rdir[:-1]+"/index.html", shell=True).wait()
        subprocess.Popen("sed -i 's/MYCONTENTS/"+mytitle+"/g' "+rdir[:-1]+"/index.html", shell=True).wait()
        subprocess.Popen("sed -i 's/MYLINK/"+pdIndex+"/g' "+rdir[:-1]+"/index.html", shell=True).wait()
    #print recoIndex
    ##make index in epoch for all reco versions
    mytitle    = "hDQM "+re.split('/',edir[:-1])[-1]
    mycontents = "hDQM Trend Plots "+re.split('/',edir[:-1])[-1]
    subprocess.Popen("cp ./test/Template_Index.html "+edir[:-1]+"/index.html", shell=True).wait()
    subprocess.Popen("sed -i 's/MYTITLE/"+mycontents+"/g' "+edir[:-1]+"/index.html", shell=True).wait()
    subprocess.Popen("sed -i 's/MYCONTENTS/"+mytitle+"/g' "+edir[:-1]+"/index.html", shell=True).wait()
    subprocess.Popen("sed -i 's/MYLINK/"+recoIndex+"/g' "+edir[:-1]+"/index.html", shell=True).wait()
#print epochIndex
subprocess.Popen("cp ./test/Template_Index.html fig/index.html", shell=True).wait()
subprocess.Popen("sed -i 's/MYTITLE/hDQM/g' fig/index.html", shell=True).wait()
subprocess.Popen("sed -i 's/MYCONTENTS/hDQM Trend Plots/g' fig/index.html", shell=True).wait()
subprocess.Popen("sed -i 's/MYLINK/"+epochIndex+"/g' fig/index.html", shell=True).wait()
####Make generic index containing all PDs for PD generation:
##To do this, I can loop over all PDs in each dir for fig/

##Copy directory structure w/figures to web area
for epoch in Epochs:
    subprocess.Popen("rm -rf "+webDir+epoch+"/*", shell=True).wait()
subprocess.Popen("cp -r fig/* "+webDir, shell=True).wait()
