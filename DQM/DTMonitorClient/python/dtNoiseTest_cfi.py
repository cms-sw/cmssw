import FWCore.ParameterSet.Config as cms

noiseTest = cms.EDAnalyzer("DTNoiseTest",
    HzThreshold = cms.untracked.int32(300),
    runningStandalone = cms.untracked.bool(True),
    meanTestName = cms.untracked.string('NoiseMeanInRange'),
    folderTag = cms.untracked.string('Occupancies'),
    folderRoot = cms.untracked.string(''),
    debug = cms.untracked.bool(False),
    diagnosticPrescale = cms.untracked.int32(1000)
)


