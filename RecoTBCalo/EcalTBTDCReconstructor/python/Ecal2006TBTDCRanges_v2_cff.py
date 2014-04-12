import FWCore.ParameterSet.Config as cms

tdcRanges = cms.VPSet(cms.PSet(
    endRun = cms.int32(14441),
    tdcMax = cms.vdouble(1008.0, 927.0, 927.0, 927.0, 927.0),
    startRun = cms.int32(10339),
    tdcMin = cms.vdouble(748.0, 400.0, 400.0, 400.0, 400.0)
), 
    cms.PSet(
        endRun = cms.int32(999999),
        tdcMax = cms.vdouble(1764.0, 927.0, 927.0, 927.0, 927.0),
        startRun = cms.int32(14442),
        tdcMin = cms.vdouble(1502.0, 400.0, 400.0, 400.0, 400.0)
    ))

