#! /usr/bin/python

import string, os, time
import commands
lumiauthpath=''
lumilogpath=''

def isCollisionRun(run,authpath=''):
    itIs = False
    itIsAlso = False
    command = 'dumpRunInfo.py -c oracle://cms_omds_lb/cms_runinfo -P '+authpath+' -r '+run+' --collision-only l1key | wc'
    statusAndOutput = commands.getstatusoutput(command)
    if statusAndOutput[1].split('   ')[2] == '2': itIs = True
    command = 'dumpRunInfo.py -c oracle://cms_omds_lb/cms_runinfo -P '+authpath+' -r '+run+' --collision-only hltkey | wc'
    statusAndOutput = commands.getstatusoutput(command)
    if statusAndOutput[1].split('   ')[2] == '2': itIsAlso = True
    return itIs and itIsAlso 

def getRunnumberFromFileName(lumifilename):
    runnumber=int(lumifilename.split('_')[4])
    return runnumber

def getRunsToBeUploaded(connectionString, dropbox, authpath=''):
    #print 'authpath ',authpath
    # get the last analyzed run
    command = 'lumiData.py -c ' +connectionString+' -P '+authpath+' --raw listrun'
    statusAndOutput = commands.getstatusoutput(command)
    lastAnalyzedRunNumber = eval(statusAndOutput[1])[-1][0]
    print 'Last analyzed run: ', lastAnalyzedRunNumber

    # check if there are new runs to be uploaded
    #command = 'ls -ltr '+dropbox
    files=filter(os.path.isfile,[os.path.join(dropbox,x) for x in os.listdir(dropbox)])
    #print files
    files.sort(key=lambda x: os.path.getmtime(os.path.join(dropbox,x)))
    #print 'sorted files ',files
    lastRaw = files[-1]
    lastRecordedRun = getRunnumberFromFileName(lastRaw)

    print 'Last lumi file produced: ', lastRaw 
    print 'Last recorded Run: ', lastRecordedRun 
	
    # if yes, fill a list with the runs yet to be uploaded
    runsToBeAnalyzed = {}
    if lastRecordedRun != lastAnalyzedRunNumber:
        #print 'latRre ',lastRecordedRun
        #print 'lastana ',lastAnalyzedRunNumber
        for file in files:
            thisrun=getRunnumberFromFileName(file)
            #print 'this run ',thisrun
            if  thisrun>lastAnalyzedRunNumber and isCollisionRun(str(thisrun),authpath): 
                    runsToBeAnalyzed[str(thisrun)] = file
    #print 'runsToBeAnalyzed ', runsToBeAnalyzed
    return runsToBeAnalyzed

import os, sys
from RecoLuminosity.LumiDB import argparse
def main():
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description="Lumi Data scan")
    parser.add_argument('-c',dest='connect',action='store',required=True,help='connect string to lumiDB')
    parser.add_argument('-d',dest='dropbox',action='store',required=True,help='location of the lumi root files')
    parser.add_argument('-norm',dest='normalization',action='store',required=True,help='lumi normalization factor')
    parser.add_argument('-P',dest='authpath',action='store',required=False,help='auth path')
    parser.add_argument('-L',dest='logpath',action='store',required=False,help='log path')
    args=parser.parse_args()
    if args.authpath:
        lumiauthpath=args.authpath
    if args.logpath:
        lumilogpath=args.logpath
        
    runsToBeAnalyzed = getRunsToBeUploaded(args.connect, args.dropbox,lumiauthpath) 

    runCounter=0
    for run in runsToBeAnalyzed:
        runCounter+=1
        if runCounter==1: print 'List of processed runs: '
        print 'Run: ', run, ' file: ', runsToBeAnalyzed[run]
        logFile=open(os.path.join(lumilogpath,'loadDB_run'+run+'.log'),'w',0)

        # filling the DB
        command = '$LOCALRT/test/$SCRAM_ARCH/loadLumiDB '+run+' "file:'+runsToBeAnalyzed[run]+'"'
        statusAndOutput = commands.getstatusoutput(command)
        logFile.write(command+'\n')
        logFile.write(statusAndOutput[1])
        if not statusAndOutput[0] == 0:
            print 'ERROR while loading info onto DB for run ' + run
            print statusAndOutput[1]
            
        # applying normalization
        command = 'applyCalibration.py -c '+args.connect+' -norm '+ args.normalization +' -r '+run+' -P '+ lumiauthpath+' run'
        statusAndOutput = commands.getstatusoutput(command)
        logFile.write(command+'\n')
        logFile.write(statusAndOutput[1])
        # apply default validation
        selectstring='"{'+run+':[]}"'
        command = 'lumiValidate.py -c '+args.connect+' -P '+ lumiauthpath+' -runls '+selectstring+' update' 
        statusAndOutput = commands.getstatusoutput(command)
        logFile.write(command+'\n')
        logFile.write(statusAndOutput[1])
        logFile.close()
        if not statusAndOutput[0] == 0:
            print 'ERROR while applying normalization to run '+ run
            print statusAndOutput[1]

    if runCounter == 0: print 'No runs to be analyzed'

if __name__=='__main__':
    main()
