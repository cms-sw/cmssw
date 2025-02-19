# The following comments couldn't be translated into the new config version:

# Configuration file for EventSetupTest_t

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.hv = cms.EDAnalyzer("DTTimeUtility",

        year   = cms.int32(2009),
        month  = cms.int32(  12),
        day    = cms.int32(  11),
        hour   = cms.int32(   0),
        minute = cms.int32(   0),
        second = cms.int32(   1),

        condTime = cms.int64(5445216652344191208),

        coralTime = cms.int64(1278584868675000000) )

process.p = cms.Path(process.hv)

