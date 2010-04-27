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
    """ compute the interval of time in seconds since the Epoch and return the packed 64bit value.
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

import os
import re
import sys

time = timeStamptoDate(long(sys.argv[1]))
print time

