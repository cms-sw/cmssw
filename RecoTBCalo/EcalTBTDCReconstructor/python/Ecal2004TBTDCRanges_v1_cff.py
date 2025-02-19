import FWCore.ParameterSet.Config as cms

tdcRanges = cms.VPSet(cms.PSet(
    endRun = cms.int32(999999),
    tdcMax = cms.vdouble(958.0, 927.0, 927.0, 927.0, 927.0),
    startRun = cms.int32(-1),
    tdcMin = cms.vdouble(430.0, 400.0, 400.0, 400.0, 400.0)
))

