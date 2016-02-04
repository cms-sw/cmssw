#!/usr/bin/env python
#GBenelli Added the /env above to use the python version of CMSSW and run without having to do python <SCRIPT NAME>

""" This script does the following:
1- reads the list of iovs (by timestamp) in a sqlite file
2- creates a cfg for each iov and runs them
3- takes the output of each job and builds a single output with the content of each iov
It is recommended to redirect the output to a file.
"""

import os
import re
import sys
import time

""" Helper functions for time conversions """

def pack(high,low):
    """pack high,low 32bit unsigned int to one unsigned 64bit long long
       Note:the print value of result number may appear signed, if the sign bit is used.
    """
    h=high<<32
    return (h|low)

def secondsFromString(i):
    """convert from a string in the format output from timeStamptoDate to a 32bit seconds from the epoch.
    The format accepted is \"DD/MM/YYYY HH:MM:SS\". The year must be the full number.
    """
    return int(time.mktime(time.strptime(i, "%d/%m/%Y %H:%M:%S")))

def packFromString(i):
    """pack from a string in the format output from timeStamptoUTC to a 64bit timestamp
    the format accepted is \"DD/MM/YYYY HH:MM:SS\" . The year must be the full number.
    """
    return pack(secondsFromString(i), 0)

def intervalSinceEpoch(i):
    """ compute the interval of time is seconds since the Epoch and return the packed 64bit value.
    """
    return( packFromString(i) - packFromString("01/01/1970 00:00:00") )

def unpack(i):
    """unpack 64bit unsigned long long into 2 32bit unsigned int, return tuple (high,low)
    """
    high=i>>32
    low=i&0xFFFFFFFF
    return(high,low)

def timeStamptoDate(i):
    """convert 64bit timestamp to local date in string format
    """
    #GBenelli Add a try: except: to handle the stop time of the last IOV "end of time"
    try:
        date=time.ctime(unpack(i)[0])
    except:
        #Handle the case of last IOV (or any IOV) timestamp being "out of range" by returning -1 instead of the date...
        print "Could not unpack time stamp %s, unpacked to %s!"%(i,unpack(i)[0])
        date=-1
    return date



# The first parameter is the name of the script
if len(sys.argv) < 3:
    print "Please provide the name of the sqlite file and the tag as in: ",
    print "./CheckAllIOVs.py Example1a.db SiStripDetVOff_Fake_31X"
    sys.exit(1)

print "Reading all IOVs"

# Example1a.db
# SiStripDetVOff_Fake_31X

database = sys.argv[1]

#GBenelli commit code from Marco to run check on a time interval:
startFrom = 0
if len(sys.argv) > 3:
    startFrom = packFromString(sys.argv[3])
endAt = 0
if len(sys.argv) > 4:
    endAt = packFromString(sys.argv[4])
    
iovs = os.popen("cmscond_list_iov -c sqlite_file:"+database+" -t "+sys.argv[2])
iovsList = iovs.read()
splittedList = re.split("payloadToken",iovsList)
splittedList = re.split("Total",splittedList[1])
splittedList = re.split("\[DB|\]\[.*\]\[.*\]\[.*\]\[.*\]", splittedList[0])
# Loop on even numbers
for i in range(0, len(splittedList), 2):
    # print splittedList[i]
    iov = re.split(" ", splittedList[i])
    if len(iov) > 1:
        start = iov[0].strip("\n")
        end = iov[2].strip("\n")
        # print "iov = ", iov
        # print "start =", start,
        # print ", end =", end

        if long(startFrom) > long(start):
            print "Skipping IOV =", start, " before requested =", startFrom
            continue
        if (endAt != 0) and (long(endAt) < long(end)):
            print "Skipping IOV =", end, " after requested =", endAt
            continue
        
        ##TODO:Should we investigate this issue? Is it going to be an issue in the DB?
        if end == "18446744073709551615":
            end = str(int(start) + 1)

        startDate = timeStamptoDate(int(start))
        endDate = timeStamptoDate(int(end))
        #GBenelli Handle here the case of "end of time" IOV end time stamp 
        if endDate==-1:
            endDate=timeStamptoDate(int(start)+1) 
            
        print "start date = ", startDate,
        print ", end date = ", endDate
        fullDates="_FROM_"+startDate.replace(" ", "_").replace(":", "_")+"_TO_"+endDate.replace(" ", "_").replace(":", "_")
        fileName="DetVOffPrint"+fullDates+"_cfg.py"
        os.system("cat templateCheckAllIOVs_cfg.py | sed -e \"s/STARTTIME/"+start+"/g\" | sed -e \"s/ENDTIME/"+end+"/g\" | sed -e \"s/DATE/"+fullDates+"/g\" | sed -e \"s/DATABASE/sqlite_file:"+database+"/g\" > "+fileName)
        # run = os.popen("cmsRun "+fileName+" > /dev/null")
        os.system("cmsRun "+fileName+" > /dev/null")

        for line in open("DetVOffReaderDebug_"+fullDates+".log", "r"):
            if "IOV" in line or "OFF" in line or "ON" in line:
                print line.strip("\n")

# # Do it afterwards because popen does not wait for the end of the job.
# for i in range(0, len(splittedList), 2):
#     iov = re.split(" ", splittedList[i])
#     if len(iov) > 1:
#         for line in open("DetVOffReaderDebug_"+fullDates+".log", "r"):
#             if "IOV" in line or "OFF" in line or "ON" in line:
#                 print line.strip("\n")
