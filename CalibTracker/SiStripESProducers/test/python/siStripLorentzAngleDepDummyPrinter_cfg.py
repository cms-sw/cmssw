# -*- coding: utf-8 -*-
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
        LorentzAngleDepReaderDebug = cms.untracked.PSet(
            threshold = cms.untracked.string('DEBUG')
        ),
        LorentzAngleDepReaderSummary = cms.untracked.PSet(
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
    input = cms.untracked.int32(10)
)
process.source = cms.Source("EmptyIOVSource",
    timetype   = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue  = cms.uint64(11),
    interval = cms.uint64(1)
)


process.load("CalibTracker.SiStripESProducers.SiStripLorentzAngleDepESProducer_cfi")

# Need to specify the Record for each BaseDelay.
# Optionally the Label associated to the tag can also be specified (default = "").

process.siStripLorentzAngleDepESProducer.LatencyRecord =   cms.PSet(
       record = cms.string('SiStripLatencyRcd'),
       label = cms.untracked.string('Latency'))
       
process.siStripLorentzAngleDepESProducer.LorentzAnglePeakMode = cms.PSet(
       record = cms.string('SiStripLorentzAngleRcd'),
       label = cms.untracked.string('LA1'))

process.siStripLorentzAngleDepESProducer.LorentzAngleDeconvMode = cms.PSet(
       record = cms.string('SiStripLorentzAngleRcd'),
       label = cms.untracked.string('LA2'))


process.poolDBESSource = cms.ESSource(
    "PoolDBESSource",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:/afs/cern.ch/user/r/rebeca/public/dbfile.db'),
    toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('SiStripLatencyRcd'),
            tag = cms.string('SiStripLatency_Ideal_31X'),
            label = cms.untracked.string('Latency')
            ),
        cms.PSet(
            record = cms.string('SiStripLorentzAngleRcd'),
            tag = cms.string('SiStripLorentzAngle_1'),
            label = cms.untracked.string('LA1')
            ),
        cms.PSet(
            record = cms.string('SiStripLorentzAngleRcd'),
            tag = cms.string('SiStripLorentzAngle_2'),
            label = cms.untracked.string('LA2')
            )
        )
    )

process.reader = cms.EDAnalyzer("SiStripLorentzAngleDepDummyPrinter")
                              
process.p1 = cms.Path(process.reader)


