#!/usr/bin/python

import re
import subprocess

##Setting variables:
#webDir = '/data/users/event_display/HDQM/Current/'
#webDir='/afs/cern.ch/user/c/cctrack/scratch0/TomislavTest/new1/CMSSW_5_2_3/src/DQM/SiStripHistoricInfoClient/test/NewHDQM/html/'
webDir= '/data/users/event_display/HDQM/dev/Tristan'
Epochs = ['Run2012A']
Recos  = ['Prompt']           ##other examples: 08Nov2011
PDs    = ['MinimumBias']      ##other examples: 'SingleMu','DoubleMu'

pwDir  = subprocess.Popen("pwd", shell=True, stdout=subprocess.PIPE).stdout.readline()[:-1]+'/'

####Build individual indices for each PD set of plots, and parent directories
epochDirs  = subprocess.Popen("ls -d1 ./fig/*", shell=True, stdout=subprocess.PIPE).stdout.readlines()
epochIndex = ''

for edir in epochDirs:
    print "SSS:",edir
    epochIndex += '<LI><A href="'+re.split('/',edir[:-1])[-1]+'">'+re.split('/',edir[:-1])[-1]+'<\\/a>\\n'
    recoDirs  = subprocess.Popen("ls -d1 "+edir[:-1]+"/*", shell=True, stdout=subprocess.PIPE).stdout.readlines()
    print "dirs:",recoDirs
    recoIndex = ''
    for rdir in recoDirs:
        recoIndex += '<LI><A href="'+re.split('/',rdir[:-1])[-1]+'">'+re.split('/',rdir[:-1])[-1]+'<\\/a>\\n'
        pdDirs     = subprocess.Popen("ls -d1 "+rdir[:-1]+"/*", shell=True, stdout=subprocess.PIPE).stdout.readlines()
        pdIndex    = ''
        for pdir in pdDirs:
            ##First setup title (PD,Epoch,Reco)
            title = re.split('/',pdir[:-1])[-1]+'_'+re.split('/',edir[:-1])[-1]+'-'+re.split('/',rdir[:-1])[-1]
            ##To run, I must be in the directory with all pngs...#
            #perlCmd = "perl "+pwDir+"test/diowroot.pl -c 2 -t "+title
            perlCmd = "perl "+pwDir+"test/diowroot_seva.pl -c 2 -t "+title
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

