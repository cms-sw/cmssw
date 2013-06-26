#!/usr/bin/python

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
    import time
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
    import time
    return time.ctime(unpack(i)[0])

""" This script does the following:
1- reads the list of iovs (by timestamp) in a sqlite file
2- creates a cfg for each iov and runs them
3- takes the output of each job and builds a single output with the content of each iov
It is recommended to redirect the output to a file.
"""

import os
import re
import sys

# The first parameter is the name of the script
if len(sys.argv) < 3:
    print "Please provide the name of the sqlite file and the tag as in: ",
    print "./CheckAllIOVs.py Example1a.db SiStripDetVOff_Fake_31X"
    sys.exit(1)

print "Reading all IOVs"

# Example1a.db
# SiStripDetVOff_Fake_31X

database = sys.argv[1]
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
        startDate = timeStamptoDate(int(start))
        endDate = timeStamptoDate(int(end))
        if end == "18446744073709551615":
            end = str(int(start) + 1)
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
