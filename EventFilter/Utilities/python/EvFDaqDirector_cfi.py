import FWCore.ParameterSet.Config as cms

EvFDaqDirector = cms.Service( "EvFDaqDirector",
    buBaseDir = cms.untracked.string(""),
    baseDir = cms.untracked.string(""),
    runNumber = cms.untracked.uint32(0)
    )

