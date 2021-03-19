import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG')
    ),
    debugModules = cms.untracked.vstring('*')
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

#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
#
#process.source = cms.Source("PoolSource",
#    # replace 'myfile.root' with the source file you want to use
#    fileNames = cms.untracked.vstring(
#        'file:myfile.root'
#    )
#)

process.load("CalibTracker.SiStripESProducers.fake.Phase2TrackerConfigurableCablingESSource_cfi")

process.demo = cms.EDAnalyzer('CheckPhase2Cabling')

process.p = cms.Path(process.demo)
