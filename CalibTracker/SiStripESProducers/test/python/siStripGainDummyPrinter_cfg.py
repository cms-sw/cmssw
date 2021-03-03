# The following comments couldn't be translated into the new config version:

# upload to database 

#string timetype = "timestamp"    

import FWCore.ParameterSet.Config as cms

process = cms.Process("Reader")

# Use this to have also debug info (WARNING: the resulting file is > 200MB.
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    debugModules = cms.untracked.vstring('*'),
    files = cms.untracked.PSet(
        GainReaderDebug = cms.untracked.PSet(
            threshold = cms.untracked.string('DEBUG')
        ),
        GainReaderSummary = cms.untracked.PSet(
            threshold = cms.untracked.string('INFO')
        )
    )
)

# How to use the EmptyIOVSource:
# the EmptyIOVSource will generate N events with a given interval.
# the N events must be specified in the maxEvents as usual but the
# first value, last value, timetype (runnumber, timestamp or lumiid) must be specified
# in the EmptyIOVSource configuration block. It will then generate events with the given
# interval.
# To generate one event per run in a given range of runs you should then use:
# - first - last value as the run range
# - interval == 1 (means move of one run unit at a time)
# - maxEvents = lastValue - firstValue (so that there is one event per run
# otherwise it will stop before completing the range or it will go beyond (to infinity).

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)
process.source = cms.Source("EmptyIOVSource",
    timetype   = cms.string('runnumber'),
    firstValue = cms.uint64(97),
    lastValue  = cms.uint64(102),
    interval = cms.uint64(1)
)

process.load("CalibTracker.SiStripESProducers.SiStripGainESProducer_cfi")

# Need to specify the Record for each ApvGain.
# Optionally the Label associated to the tag can also be specified (default = "").
process.siStripGainESProducer.APVGain = cms.VPSet(
   cms.PSet(
       Record = cms.string('SiStripApvGainRcd'),
       Label = cms.untracked.string('SiStripApvGain_test_1')
   ),
   # cms.PSet(
   #     Record = cms.string('SiStripApvGain2Rcd'),
   #     Label = cms.untracked.string('SiStripApvGain_test_2')
   cms.PSet(
       Record = cms.string('SiStripApvGain2Rcd'),
   ),
)

# process.siStripGainESProducer.APVGain[0].Label = "SiStripApvGain_test_1"

# From CondCore/ESSources V09-00-06 it is possible to use a single ESSource.
# For earlier versions the two tags to go in the same record must be loaded by two different PoolDBESSources.
process.poolDBESSource = cms.ESSource(
    "PoolDBESSource",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:dbfile.db'),
    toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('SiStripApvGainRcd'),
            tag = cms.string('SiStripApvGain_test_1'),
            label = cms.untracked.string('SiStripApvGain_test_1')
        ),
#        cms.PSet(
#            record = cms.string('SiStripApvGainRcd'),
#            tag = cms.string('SiStripApvGain_test_2'),
#            label = cms.untracked.string('SiStripApvGain_test_2')
#        )
    )
)

process.poolDBESSource2 = cms.ESSource("PoolDBESSource",
   BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
   DBParameters = cms.PSet(
       messageLevel = cms.untracked.int32(2),
       authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
   ),
   timetype = cms.untracked.string('runnumber'),
   connect = cms.string('sqlite_file:dbfile.db'),
   toGet = cms.VPSet(cms.PSet(
       record = cms.string('SiStripApvGain2Rcd'),
       # tag = cms.string('SiStripApvGain_test_2'),
       # label = cms.untracked.string('SiStripApvGain_test_2')
       tag = cms.string('SiStripApvGain_Ideal_31X'),
       # label = cms.untracked.string('')
   ))
)

process.reader = cms.EDFilter("SiStripGainDummyPrinter")
                              
process.p1 = cms.Path(process.reader)


