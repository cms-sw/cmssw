#!/usr/bin/env python

import os,sys,time

#LUMICSV="lumi_until1816.txt"
#LUMICSV="lumi_until1823.txt"
LUMICSV="/afs/cern.ch/cms/lumi/www/publicplots/totallumivstime-2011.csv"
RUNFILL="/afs/cern.ch/cms/CAF/CMSCOMM/COMM_GLOBAL/lhcfiles_prod/runtofill_dqm.txt"
LPCFILES='/afs/cern.ch/cms/CAF/CMSCOMM/COMM_GLOBAL/LHCFILES'

lumirun={}
lumifill={}

runincsv=[]
runinmap=[]

totdeliv=0.
print "######### checking lumi ################################################"
print LUMICSV
print LPCFILES
print "########################################################################"
lumi=open(LUMICSV,'r')
for line in lumi.readlines():
    linesplit=line.split(',')
    if len(linesplit)==5:
        if not linesplit[0].isdigit(): continue # skip comments lines
        runno=int(linesplit[0])
        delivered=float(linesplit[3])
        recorded=float(linesplit[4])
        lumirun[runno]=delivered
        totdeliv+=delivered
        runincsv.append(runno)
lumi.close()


# doing fill-by-fill
r_f=open(RUNFILL,'r')
for line in r_f.readlines():
    linesplit=line.split()
    if len(linesplit)!=2: continue
    if not linesplit[0].isdigit(): continue # skip comments and LASTFILL lines
    runno=int(linesplit[0])
    fillno=int(linesplit[1])
    runinmap.append(runno)
    
    if runno not in lumirun.keys():
#        print "Warning, run not in lumi csv file: ",runno
        continue
    if fillno in lumifill.keys():
        lumifill[fillno]+=lumirun[runno]
    else:
        lumifill[fillno]=lumirun[runno]
r_f.close()

for run in runinmap:
    if run not in runincsv:
        print "run in map but not in cvs:",run
for run in runincsv:
    if run not in runinmap:
        if lumirun[run]>0.:
            print "run in csv but not in map (probably DAQ was not running!):%s %6.2f" %(run,lumirun[run]/1000000.)
        

# now read summary files in LPC format

filllist=lumifill.keys()
filllist.sort()
tot_lumi1=0.
tot_lumi2=0.

for fill in filllist:
    # build filename
    fillsummary=LPCFILES+'/'+str(fill)+'/'+str(fill)+'_summary_CMS.txt'
    intlumi=0.
    #    print fillsummary
    if os.path.exists(fillsummary):
        fs=open(fillsummary,'r')
        for line in fs.readlines():
            linesplit=line.split()
    #        print linesplit
            if len(linesplit)!=4: continue
            intlumi+=float(linesplit[3])
        fs.close()
    lumi1=lumifill[fill]/1000000.
    lumi2=intlumi/1000000.
    diff=lumi1-lumi2
    if lumi2!=0.:
        rel=1-lumi1/lumi2
    else:
        rel=-1.
    print "fill nr: %d csv= %6.2f lpc= %6.2f diff= %6.2f rel= %6.2f" % (fill,lumi1,lumi2,diff,rel)
    tot_lumi1+=lumi1
    tot_lumi2+=lumi2
    
tot_diff=tot_lumi1-tot_lumi2
tot_rel=1-tot_lumi1/tot_lumi2

print "-----------------------------------------------------------------------------------"
print "tot runinmap:  csv= %6.2f lpc= %6.2f diff= %6.2f rel= %6.2f" % (tot_lumi1,tot_lumi2,tot_diff,tot_rel)
print "tot onlycsv :  csv= %6.2f" % (totdeliv/1000000.)

    

