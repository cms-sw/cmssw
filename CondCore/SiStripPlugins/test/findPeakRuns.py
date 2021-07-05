#!/usr/bin/env python3

'''
Script used to find all Strip Tracker PEAK runs

Usage:

 create the list of SiStripLatency::singleReadOutMode() IOVs / values with:

 getPayloadData.py --plugin pluginSiStripLatency_PayloadInspector --plot plot_SiStripIsPeakModeHistory --tag SiStripLatency_GR10_v2_hlt --time_type Run --iovs '{"start_iov": "1", "end_iov": "400000"}' --db Prod --test > & log.json

 create the list of runs with the Strip Tracker in global DAQ

 getPayloadData.py --plugin pluginRunInfo_PayloadInspector --plot plot_RunInfoTrackerHistory --tag runinfo_31X_hlt --time_type Run --iovs '{"start_iov": "1", "end_iov": "400000"}' --db Prod --test > & isTrackerin.json

 followed by:

 python findPeakRuns.py -f log.json -r isTrackerIn.json > & allTrackerRuns.csv &
'''

from __future__ import print_function
import json
import ROOT
from pprint import pprint
from optparse import OptionParser
import CondCore.Utilities.conddblib as conddb

##############################################
def findPeakIOVs(values):
##############################################
    listOfIOVs=[]
    listOfModes=[]
    lastMode=1
    lastRun=1
    
    latestRun=-1
    latestMode=-9999

    for value in values:
        if (value['y']!=lastMode):
            listOfIOVs.append((lastRun,value['x']))
            if(lastMode==1):
                listOfModes.append('PEAK')
            else:
                listOfModes.append('DECO')
            lastMode=value['y']
            lastRun=value['x']
        else:
            latestRun=value['x']
            if(value['y']==1):
                latestMode='PEAK'
            else:
                latestMode='DECO'

    ## special case for the last open IOV
    listOfIOVs.append((lastRun,999999))
    listOfModes.append(latestMode)

    return dict(zip(listOfIOVs,listOfModes))

##############################################
def getTkInDict(runvalues):
##############################################
    isTrackerIn = {}
    for entry in runvalues:
        isTrackerIn[entry['x']]=entry['y']
    return isTrackerIn

##############################################
def getAllRuns():
##############################################
    con = conddb.connect(url = conddb.make_url("pro"))
    session = con.session()
    RUNINFO = session.get_dbtype(conddb.RunInfo)
    allRuns = session.query(RUNINFO.run_number, RUNINFO.start_time, RUNINFO.end_time).all()
    return allRuns

##############################################
def convert_timedelta(duration):
##############################################
    days, seconds = duration.days, duration.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = (seconds % 60)
    return '{0:02}'.format(hours), '{0:02}'.format(minutes), '{0:02}'.format(seconds)

##############################################
def main():
##############################################

    parser = OptionParser()
    parser.add_option("-f", "--file", dest="filename",
                      help="open FILE and extract info", metavar="FILE")
    parser.add_option("-r", "--runfile", dest="runfilename",
                      help="open RUNFILE and extract info", metavar="RUNFILE")
    parser.add_option("-v", "--verbose",
                      action="store_true", dest="verbose", default=False,
                      help="verbose output")
    parser.add_option("-p", "--peakOnly",
                      action="store_true", dest="peakOnly", default=False,
                      help="Print only runs in PEAK mode")
    (options, args) = parser.parse_args()

    with open(options.filename) as data_file:
        data = json.load(data_file)
        values = data["data"]
    
    IOVs = findPeakIOVs(values)
    if(options.verbose):
        pprint(IOVs)

    with open(options.runfilename) as rundata_file:
        data = json.load(rundata_file)
        runvalues = data["data"]

    isTrackerIn = getTkInDict(runvalues)

    if(options.verbose):
        for value in runvalues:
            isTrackerIn = bool(value['y'])
            runnumber = int(value['x'])
            if(not isTrackerIn):
                # there was no Tracker in this run
                continue
            else:
                for key, value in IOVs.items():
                    if(key[0]<runnumber and key[1]>runnumber):
                        print(runnumber,",",key[0],"-",key[1],",",value)

    allRuns = getAllRuns()
    if(options.verbose):
        print(allRuns)

    sorted_runs = sorted(allRuns)

    for run in sorted_runs:
        if run[0] not in isTrackerIn:
            continue
        if(isTrackerIn[run[0]]):
            #print(run[0],"Tracker In")
            for key, value in IOVs.items():
                if(key[0]<run[0] and key[1]>run[0]):
                    hours, minutes, seconds = convert_timedelta(run[2]-run[1])
                    if(options.peakOnly):
                        if(value=='PEAK'):
                            print(run[0],",",run[1].strftime('%Y-%m-%d'),",",'{}:{}:{}'.format(hours,minutes,seconds),",",key[0],"-",key[1],",",value)
                        else:
                            pass
                    else:
                        print(run[0],",",run[1].strftime('%Y-%m-%d'),",",'{}:{}:{}'.format(hours,minutes,seconds),",",key[0],"-",key[1],",",value)
        else:
            pass
            #print(run[0],"Tracker Out")

if __name__ == "__main__":
    main()
