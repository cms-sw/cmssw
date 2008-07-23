import FWCore.ParameterSet.Config as cms

EcalTBSimTDCRanges = cms.PSet(
tdcRanges = cms.VPSet(cms.PSet(
    endRun = cms.int32(999999),
    tdcMax = cms.vdouble(1008.0, 927.0, 927.0, 927.0, 927.0),
    startRun = cms.int32(-1),
    tdcMin = cms.vdouble(748.0, 400.0, 400.0, 400.0, 400.0)
))
)
