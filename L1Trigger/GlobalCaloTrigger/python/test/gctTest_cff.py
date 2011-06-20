import FWCore.ParameterSet.Config as cms
    
gctemu = cms.EDFilter("L1GctTest",
    doElectrons = cms.untracked.bool(False),
    doSingleEvent = cms.untracked.bool(False),
    doEnergyAlgos = cms.untracked.bool(False),
    doFirmware = cms.untracked.bool(False),
    doRealData = cms.untracked.bool(False),
    useNewTauAlgo = cms.untracked.bool(False),
    printConfig = cms.untracked.bool(False),
    inputFile = cms.untracked.string(''),
    energySumsFile = cms.untracked.string(''),
    referenceFile = cms.untracked.string(''),
    preSamples = cms.uint32(2),
    postSamples = cms.uint32(2)
)

