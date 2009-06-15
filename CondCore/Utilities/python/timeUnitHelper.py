"""Some helper classes to convert conditions time units back and forth
"""
def pack(high,low):
    """pack high,low 32bit unsigned int to one unsigned 64bit long long
       Note:the print value of result number may appear signed, if the sign bit is used.
    """
    h=high<<32
    return (h|low)

def packToString(high,low):
    """pack high,low 32bit unsigned int to one unsigned 64bit long long in string format
       Note:the print value of result number may appear signed, if the sign bit is used.
    """
    fmt="%u"
    return fmt%pack(high,low)

def unpack(i):
    """unpack 64bit unsigned long long into 2 32bit unsigned int, return tuple (high,low)
    """
    high=i>>32
    low=i&0xFFFFFFFF
    return(high,low)

def unpackFromString(i):
    """unpack 64bit unsigned long long in string format into 2 32bit unsigned int, return tuple(high,low)
    """
    return unpack(int(i))

def timeStamptoDate(i):
    """convert 64bit timestamp to local date in string format
    """
    import time
    return time.ctime(unpack(i)[0])

def timeStamptoUTC(i):
    """convert 64bit timestamp to Universal Time in string format
    """
    t=unpack(i)[0]
    import time
    return time.strftime("%a, %d %b %Y %H:%M:%S +0000",time.gmtime(t))
                         
def unpackLumiid(i):
    """unpack 64bit lumiid to dictionary {'run','lumisection'}
    """
    j=unpack(i)
    return {'run':j[0],'lumisection':j[1]}
