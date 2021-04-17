# This configuration tests CondDBESSource and the EventSetup
# system. It causes it to read BeamSpot data objects from
# the database. The global tag and run number are selected
# to access this data in a run which is known to have
# multiple IOVs in the same run. This is a very simple job
# with empty input and one event per luminosity block.
# The analyzer gets this BeamSpot object and prints
# values from it to standard out. This is compared to
# a reference file of known expected files. If the
# correct data is not read and written to output then
# the unit test fails. The additional wrinkle here is that
# the analyzer puts in a "busy wait" (intentional does
# some calculation that uses CPU for a while). It does
# this for each event that is the first event of a
# new IOV.  That forces multiple IOVs to be running
# concurrently. Further the delay decreases as we move
# forward so the data gets are not occurring in the same
# order as the lumis are being input. The purpose of
# all this is to test that CondDBESSource and the
# EventSetup system can process IOVs concurrently
# and deliver the same data objects as when processing
# one IOV at a time.

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

#process.Tracer = cms.Service("Tracer")

process.options = dict(
    numberOfThreads = 4,
    numberOfStreams = 4,
    numberOfConcurrentRuns = 1,
    numberOfConcurrentLuminosityBlocks = 4,
    eventSetup = dict(
        numberOfConcurrentIOVs = 4
    )
)

process.load("FWCore.MessageService.MessageLogger_cfi")
#process.MessageLogger.cerr.threshold = 'INFO'
#process.MessageLogger.cerr.INFO.limit = 1000000

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(132598),
    firstLuminosityBlock = cms.untracked.uint32(1),
    firstEvent = cms.untracked.uint32(1),
    numberEventsInRun = cms.untracked.uint32(1000000),
    numberEventsInLuminosityBlock = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(200)
)

process.GlobalTag = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string(''),
        authenticationSystem = cms.untracked.int32(0),
        messageLevel = cms.untracked.int32(0),
        security = cms.untracked.string('')
    ),
    DumpStat = cms.untracked.bool(False),
    ReconnectEachRun = cms.untracked.bool(False),
    RefreshAlways = cms.untracked.bool(False),
    RefreshEachRun = cms.untracked.bool(False),
    RefreshOpenIOVs = cms.untracked.bool(False),
    connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
    globaltag = cms.string(''),
    pfnPostfix = cms.untracked.string(''),
    pfnPrefix = cms.untracked.string(''),
    snapshotTime = cms.string('2020-10-10 00:00:00.000'),
    toGet = cms.VPSet(cms.VPSet(cms.PSet(record = cms.string("BeamSpotObjectsRcd"),
                                         tag = cms.string("BeamSpotObjects_2017UL_LumiBased_v2")
                                         ))
    )
)

process.test = cms.EDAnalyzer("TestConcurrentIOVsCondCore")

process.p = cms.Path(process.test)
