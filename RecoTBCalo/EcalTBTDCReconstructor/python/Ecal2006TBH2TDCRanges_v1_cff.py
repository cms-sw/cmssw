import FWCore.ParameterSet.Config as cms

tdcZeros = cms.VPSet(cms.PSet(
    endRun = cms.int32(31031),
    tdcZero = cms.double(1050.5),
    startRun = cms.int32(27540)
), 
    cms.PSet(
        endRun = cms.int32(999999),
        tdcZero = cms.double(1058.5),
        startRun = cms.int32(31032)
    ))

