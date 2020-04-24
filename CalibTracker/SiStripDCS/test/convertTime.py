#!/usr/bin/env python

import sys
import time
import calendar

""" Converts between a 64bit timestamp and a human readable string
usage: ./convertTime.py [-l] time1 [time2 ...] 
   - "-l" to use local time
   - "time" is either a  64bit timestamp or a string formatted "DD/MM/YYYY HH:MM:SS"
"""


def pack(high,low):
    """pack high,low 32bit unsigned int to one unsigned 64bit long long
       Note:the print value of result number may appear signed, if the sign bit is used.
    """
    h=high<<32
    return (h|low)

def secondsFromString(t, localTime = True):
    """convert from a string in the format output from timeStamptoDate to a 32bit seconds from the epoch.
    If the time is UTC, the boolean value localTime must be set to False.
    The format accepted is \"DD/MM/YYYY HH:MM:SS\". The year must be the full number.
    """
    # time string, format -> time structure
    timeStruct = time.strptime(t, "%d/%m/%Y %H:%M:%S")
    if localTime:
        # time structure -> timestamp float -> timestamp int
        return int(time.mktime(timeStruct))
    else:
        # time structrue -> timestamp int
        return calendar.timegm(timeStruct)

def packFromString(s, localTime = True):
    """pack from a string in the format output from timeStamptoDate to a 64bit timestamp.
    If the time is UTC, the boolean value localTime must be set to False.
    The format accepted is \"DD/MM/YYYY HH:MM:SS\" . The year must be the full number.
    """
    return pack(secondsFromString(s, localTime), 0)
    
def unpack(i):
    """unpack 64bit unsigned long long into 2 32bit unsigned int, return tuple (high,low)
    """
    high=i>>32
    low=i&0xFFFFFFFF
    return(high,low)
    
def addZeros(time):
    """Adds a zero to the start of a single digit number"""
    timeString = str(time)
    if len(timeString) < 2:
        return ("0"+timeString)
    return timeString
    
def getMonth(s):
        months = { 'Jan':1, 'Feb':2, 'Mar':3, 'Apr': 4, 'May': 5, 'Jun': 6,
                   'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12 }
        return months[s]

def timeStamptoDate(i, localTime = True):
    """convert 64bit timestamp to local date in string format.
    If the time is UTC, the boolean value localTime must be set to False.
    The format accepted is \"DD/MM/YYYY HH:MM:SS\" . The year must be the full number.
    """
    #GBenelli Add a try: except: to handle the stop time of the last IOV "end of time"
    try:
        if localTime:
            # 64bit timestamp -> 32bit timestamp(high) -> timestamp string (local)
            date=time.ctime(unpack(i)[0])
        else:
            # 64bit timestamp -> 32bit timestamp(high) -> time tuple -> timestamp string (UTC)
            date=time.asctime(time.gmtime(unpack(i)[0]))
        # change date to "DD/MM/YYYY HH:MM:SS" format
        date = date.split()
        date[1] = getMonth(date[1])
        date = addZeros(date[2]) +'/'+ addZeros(date[1]) +'/'+  date[4] +' '+ date[3]
    except:
        #Handle the case of last IOV (or any IOV) timestamp being "out of range" by returning -1 instead of the date...
        print "Could not unpack time stamp %s, unpacked to %s!"%(i,unpack(i)[0])
        date=-1
    return date
    
def printUsage():
    print 'usage: ./convertTime.py time localTime'
    print '   - "time" is either a  64bit timestamp or a string formatted "DD/MM/YYYY HH:MM:SS"'
    print '   - "useUTC" is a bool that defaults to True (set to False for local time)'



def main(time, localTime=True):
    # convert 64bit timestamp to time string
    if time.isdigit():
        time = long(time)
        return timeStamptoDate(time, localTime)
        
    # convert time string to 64bit timestamp
    else:
        return packFromString(time, localTime)
    


if __name__ == "__main__":
    args = sys.argv[:] 
    if len(args) < 2 :
        printUsage()
        sys.exit(1)
    args = args[1:]
    if args[0]=='-h' or args[0]=='--help':
        printUsage()
        sys.exit(0)
        args = args[1:]
        
    
    useUTC = True
    if args[0]=='-l' or args[0]=='--localtime': 
      useUTC = False
      args=args[1:]
    
    for time0 in args:
      time1 = main(time0, not useUTC)
      print time0, '->', time1




