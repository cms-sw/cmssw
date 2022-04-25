import FWCore.ParameterSet.Config as cms

cepgenOutputModules = cms.PSet(
    dump = cms.PSet(
        printEvery = cms.uint32(1000),
    )
)
