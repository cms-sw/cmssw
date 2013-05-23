#!/usr/bin/env python

""" This script can be used to run manually the dcs o2o over an interval of time
dividing the it in smaller intervals. Running on an interval too big can cause
a result of the query to the database so big that the machine runs out of memory.
By splitting it in smaller intervals of a given DeltaT it is possible to keep
under control the memory used.
"""

import sys
import time
import os

"""Time utilities"""

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

"""Helper functions"""

def addZeros(time):
    """Adds a zero to the start of a single digit number"""
    timeString = str(time)
    if len(timeString) < 2:
        return ("0"+timeString)
    return timeString

def extractDate(timeString):
    """Returns the date in string format"""
    tokens = timeString.split("(")[1].split(",")
    year = tokens[0].strip()
    month = tokens[1].strip()
    day = tokens[2].strip()
    hour = tokens[3].strip()
    minute = tokens[4].strip()
    second = tokens[5].strip()
    microsecond = tokens[6].split(")")[0].strip()
    # print year, month, day, hour, minute, second, microsecond
    # return (year, month, day, hour, minute, second, microsecond)
    return ( str(day)+"/"+str(month)+"/"+str(year)+" "+str(addZeros(hour))+":"+str(minute)+":"+str(addZeros(second)) )

def extractTime(timeName):
    """This function extracts the time corresponding to the
    specified string from the configuration file and returs
    a tuple with the date
    """
    # First loop to extract the starting and ending time
    inputFile = open("dcs_o2o_template_cfg.py", "r")
    for line in inputFile:
        if timeName in line and not line.startswith("#") and not "DeltaTmin" in line:
            # Tmin = cms.vint32(2009, 12, 1,  8,  0, 0, 000),
            # print line.split("(")
            timeString = line.split("=")[1].strip()
            return extractDate(timeString)


# deltaT in hours
deltaT = 1
if len(sys.argv) > 1:
    deltaT = sys.argv[1]

deltaTinSeconds = intervalSinceEpoch( "01/01/1970 "+str(addZeros(deltaT))+":00:00" )

print "Splitting to intervals of", deltaT, "hour"

tMin = extractTime("Tmin")
print "tMin = ", tMin
tMinPacked = packFromString(tMin)

tMax = extractTime("Tmax")
print "tMax = ", tMax
tMaxPacked = packFromString(tMax)

newTmax = tMinPacked+deltaTinSeconds

while newTmax <= tMaxPacked:
    # print "new Tmax = ", newTmax
    parsed = time.strptime(timeStamptoDate(newTmax))

    # Tmin = cms.vint32(2009, 12, 1,  8,  0, 0, 000),
    year = time.strftime("%Y", parsed)
    month = int(time.strftime("%m", parsed))
    day = time.strftime("%e", parsed)
    hour = int(time.strftime("%H", parsed))
    minute = int(time.strftime("%M", parsed))
    second = int(time.strftime("%S", parsed))

    newTmaxString = " = cms.vint32("+year +","+ str(month) +","+ day +","+ str(hour) +","+ str(minute) +","+ str(second) + ", 000),"

    print "Creating cfg with tMax = ", newTmaxString

    newTmaxDateString = year.strip()+"_"+str(month).strip()+"_"+str(day).strip()+"_"+str(hour).strip()+"_"+str(minute).strip()+"_"+str(second).strip()

    cfgName = "dcs_o2o_"+str(newTmaxDateString)+"_cfg.py"
    outputFile = open(cfgName, "w")
    inputFile = open("dcs_o2o_template_cfg.py", "r")
    for line in inputFile:
        if "Tmax" in line and not line.startswith("#"):
            firstPart = line.split("=")[0]
            outputFile.write(firstPart + newTmaxString)
        else:
            outputFile.write(line)

    outputFile.close()

    os.system("cmsRun "+cfgName)

    newTmax += deltaTinSeconds
