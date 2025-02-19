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

import FWCore.ParameterSet.Config as cms

process = cms.Process("Reader")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring("*"),
    DetVOffReaderSummary_DATE = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    DetVOffReaderDebug_DATE = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    destinations = cms.untracked.vstring('DetVOffReaderSummary_DATE', 'DetVOffReaderDebug_DATE')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# Check
# print "converting start date = 28/07/2009 08:53:53 to ",
# print packFromString("28/07/2009 08:53:53")
# print "converting end date = 28/07/2009 14:13:31 to ",
# print packFromString("28/07/2009 14:13:31")
print "using an interval of 1 second = ",
print intervalSinceEpoch("01/01/1970 00:00:01")

process.source = cms.Source("EmptyIOVSource",
    timetype   = cms.string('timestamp'),
    # firstValue = cms.uint64(packFromString("28/07/2009 10:53:53")),
    # lastValue = cms.uint64(packFromString("28/07/2009 16:13:31")),
    firstValue = cms.uint64(STARTTIME),
    lastValue  = cms.uint64(ENDTIME),

    # One second inverval
    interval = cms.uint64(intervalSinceEpoch("01/01/1970 00:00:01"))
)

process.poolDBESSource = cms.ESSource("PoolDBESSource",
   BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
   DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    connect = cms.string('DATABASE'),
    toGet = cms.VPSet(cms.PSet(
    timetype = cms.untracked.string('timestamp'),
        record = cms.string('SiStripDetVOffRcd'),
        tag = cms.string('SiStripDetVOff_Fake_31X')
    ))
)

process.reader = cms.EDFilter("SiStripDetVOffDummyPrinter")
                              
process.p1 = cms.Path(process.reader)


