#!/usr/bin/env python

import string, os, time,re
import commands
lumiauthpath=''
lumilogpath=''
loaderconf=''

#def isCollisionRun(run,authpath=''):
#    itIs = False
#    itIsAlso = False
#    isInAfill = False
#    command = 'dumpRunInfo.py -c oracle://cms_omds_lb/cms_runinfo -P '+authpath+' -r '+run+' --collision-only l1key | wc'
#    statusAndOutput = commands.getstatusoutput(command)
#    if statusAndOutput[1].split('   ')[2] == '2': itIs = True
#    command = 'dumpRunInfo.py -c oracle://cms_omds_lb/cms_runinfo -P '+authpath+' -r '+run+' --collision-only hltkey | wc'
#    statusAndOutput = commands.getstatusoutput(command)
#    if statusAndOutput[1].split('   ')[2] == '2': itIsAlso = True
#    command = 'dumpRunInfo.py -c oracle://cms_omds_lb/cms_runinfo -P '+authpath+' -r '+run+' fill'
#    statusAndOutput = commands.getstatusoutput(command)
#    fillnum=statusAndOutput[1].split('\n')[1].split(' ')[1]
#    if fillnum and fillnum != '0':
#        isInAfill=True
#    return itIs and itIsAlso and isInAfill

def getRunnumberFromFileName(lumifilename):
    runnumber=int(lumifilename.split('_')[4])
    return runnumber

def getRunsToBeUploaded(connectionString, dropbox, authpath='',minrun=180250):
    #print 'authpath ',authpath
    # get the last analyzed run
    command = 'lumiData2.py -c ' +connectionString+' -P '+authpath+' listrun'
    if minrun:
        command+=' --minrun '+str(minrun)
    statusAndOutput = commands.getstatusoutput(command)
    rlist= eval(statusAndOutput[1])
    if rlist:
        lastAnalyzedRunNumber = rlist[-1]
        print 'Last run in DB: ', lastAnalyzedRunNumber
    else:
        print 'No qualified run found in DB'
        lastAnalyzedRunNumber=int(minrun)
    # check if there are new runs to be uploaded
    #command = 'ls -ltr '+dropbox
    p=re.compile('^CMS_LUMI_RAW_\d\d\d\d\d\d\d\d_\d\d\d\d\d\d\d\d\d_\d\d\d\d_\d.root$')
    files=list(filter(os.path.isfile,[os.path.join(dropbox,x) for x in os.listdir(dropbox) if p.match(x)]))
    files.sort(key=lambda x: os.path.getmtime(os.path.join(dropbox,x)))
    #print 'sorted files ',files
    #print files
    #print qualifiedfiles
    lastRaw=files[-1]
    lastRecordedRun = getRunnumberFromFileName(lastRaw)
    print 'Last lumi file produced by HF: ', lastRaw +', Run: ', lastRecordedRun 
	
    # if yes, fill a list with the runs yet to be uploaded
    runsToBeAnalyzed = {}
    if lastRecordedRun != lastAnalyzedRunNumber:
        for file in files:
            if len(file.split('_'))!=7: continue
            thisrun=getRunnumberFromFileName(file)
            #print 'this run ',thisrun,lastAnalyzedRunNumber
            #if  thisrun>lastAnalyzedRunNumber and isCollisionRun(str(thisrun),authpath):
            if thisrun>lastAnalyzedRunNumber :
                runsToBeAnalyzed[str(thisrun)] = file
    return runsToBeAnalyzed

import os, sys
from RecoLuminosity.LumiDB import argparse
def main():
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description="Lumi Data scan")
    parser.add_argument('-c',dest='connect',action='store',required=True,help='connect string to lumiDB')
    parser.add_argument('-d',dest='dropbox',action='store',required=True,help='location of the lumi root files')
    parser.add_argument('-P',dest='authpath',action='store',required=False,help='auth path')
    parser.add_argument('-L',dest='logpath',action='store',required=False,help='log path')
    parser.add_argument('-f',dest='loaderconf',action='store',required=True,help='path to loder config file')
    parser.add_argument('--minrun',dest='minrun',action='store',required=False,help='minimum run to serch')
    args=parser.parse_args()
    if args.authpath:
        lumiauthpath=args.authpath
    if args.logpath:
        lumilogpath=args.logpath
    loaderconf=args.loaderconf
    runsToBeAnalyzed = getRunsToBeUploaded(args.connect,args.dropbox,lumiauthpath,minrun=args.minrun) 
    
    runCounter=0
    rs=sorted(runsToBeAnalyzed.keys())
    for run in rs:
        runCounter+=1
        if runCounter==1: print 'List of processed runs: '
        print 'Run: ', run, ' file: ', runsToBeAnalyzed[run]
        logFile=open(os.path.join(lumilogpath,'loadDB_run'+run+'.log'),'w',0)

        # filling the DB
        command = '$LOCALRT/test/$SCRAM_ARCH/cmmdLoadLumiDB -r '+run+' -L "file:'+runsToBeAnalyzed[run]+'"'+' -f '+loaderconf+' --debug'
        statusAndOutput = commands.getstatusoutput(command)
        logFile.write(command+'\n')
        logFile.write(statusAndOutput[1])
        if not statusAndOutput[0] == 0:
            print 'ERROR while loading info onto DB for run ' + run
            print statusAndOutput[1]
            
    #    selectstring='"{'+run+':[]}"'
    #    command = 'lumiValidate.py -c '+args.connect+' -P '+ lumiauthpath+' -runls '+selectstring+' update' 
    #    statusAndOutput = commands.getstatusoutput(command)
    #    logFile.write(command+'\n')
    #    logFile.write(statusAndOutput[1])
    #    logFile.close()
    #    if not statusAndOutput[0] == 0:
    #        print 'ERROR while applying validation flag to run '+ run
    #        print statusAndOutput[1]
    if runCounter == 0: print 'No runs to be analyzed'

if __name__=='__main__':
    main()
