#! /usr/bin/python

import string, os, time
import commands

def isCollisionRun(run):
    itIs = False
    command = 'dumpRunInfo.py -c oracle://cms_omds_lb/cms_runinfo -P /nfshome0/hcallumipro  -r '+run+' --collision-only l1key | wc'
    statusAndOutput = commands.getstatusoutput(command)
    if statusAndOutput[1].split('   ')[2] == '2': itIs = True
    return itIs


def getRunsToBeUploaded(connectionString, dropbox):
    # get the last analyzed run
    command = 'lumiData.py -c "' +connectionString+'" -P /nfshome0/hcallumipro/ --raw listrun'
    statusAndOutput = commands.getstatusoutput(command)
    lastAnalyzedRunNumber = eval(statusAndOutput[1])[-1][0]
    print 'Last analyzed run: ', lastAnalyzedRunNumber

    # check if there are new runs to be uploaded
    command = 'ls -ltr '+dropbox
    for file in os.popen(command):
        lastRaw = file[0:len(file)-1]
        lastRecordedRun = lastRaw[len(lastRaw)-18:len(lastRaw)-12] #this is weak

    print 'Last lumi file produced: ', lastRecordedRun 

    # if yes, fill a list with the runs yet to be uploaded
    runsToBeAnalyzed = {}
    if int(lastRecordedRun) != lastAnalyzedRunNumber:
        fillRunsToBeAnalyzedPool = False
        for file in os.popen(command):
            lastRaw = file[0:len(file)-1]
            if str(lastAnalyzedRunNumber) == lastRaw[len(lastRaw)-18:len(lastRaw)-12]:
                fillRunsToBeAnalyzedPool = True
                continue
            if fillRunsToBeAnalyzedPool:
                if isCollisionRun(lastRaw[len(lastRaw)-18:len(lastRaw)-12]): 
                    runsToBeAnalyzed[lastRaw[len(lastRaw)-18:len(lastRaw)-12]] = lastRaw[lastRaw.find('CMS'):]

    return runsToBeAnalyzed


import os, sys
from RecoLuminosity.LumiDB import argparse
def main():
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description="Lumi Data scan")
    parser.add_argument('-c',dest='connect',action='store',required=True,help='connect string to lumiDB')
    parser.add_argument('-d',dest='dropbox',action='store',required=True,help='location of the lumi root files')
    parser.add_argument('-norm',dest='normalization',action='store',required=True,help='lumi normalization factor')
    args=parser.parse_args()

    runsToBeAnalyzed = getRunsToBeUploaded(args.connect, args.dropbox) 

    runCounter=0
    for run in runsToBeAnalyzed:
        runCounter+=1
        if runCounter==1: print 'List of processed runs: '
        print 'Run: ', run, ' file: ', runsToBeAnalyzed[run]
        logFile=open(os.path.join('/nfshome0/hcallumipro/testLog/','loadDB_run'+run+'.log'),'w',0)

        # filling the DB
        command = '$LOCALRT/test/$SCRAM_ARCH/loadLumiDB '+run+' "file:/dropbox/hcallumipro/'+runsToBeAnalyzed[run]+'"'
        statusAndOutput = commands.getstatusoutput(command)
        logFile.write(statusAndOutput[1])
        if not statusAndOutput[0] == 0:
            print 'ERROR while loading info onto DB for run ' + run
            print statusAndOutput[1]
            
        # applying normalization
        command = 'applyCalibration.py -c '+args.connect+' -norm '+ args.normalization +' -r '+run+' -P /nfshome0/hcallumipro/ run'
        statusAndOutput = commands.getstatusoutput(command)
        logFile.write(statusAndOutput[1])
        logFile.close()
        if not statusAndOutput[0] == 0:
            print 'ERROR while applying normalization to run '+ run
            print statusAndOutput[1]

    if runCounter == 0: print 'No runs to be analyzed'

if __name__=='__main__':
    main()
