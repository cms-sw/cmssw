import FWCore.ParameterSet.Config as cms

tdcRanges = cms.VPSet(cms.PSet(
    endRun = cms.int32(99999),
    tdcMax = cms.vdouble(1531.0, 927.0, 927.0, 927.0, 927.0),
    startRun = cms.int32(16585),
    tdcMin = cms.vdouble(1269.0, 400.0, 400.0, 400.0, 400.0)
))

