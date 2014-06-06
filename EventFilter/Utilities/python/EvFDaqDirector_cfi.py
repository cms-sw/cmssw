import FWCore.ParameterSet.Config as cms

EvFDaqDirector = cms.Service( "EvFDaqDirector",
    buBaseDir = cms.untracked.string(options.buBaseDir),
    baseDir = cms.untracked.string(options.dataDir),
    runNumber = cms.untracked.uint32(options.runNumber)
    )

