import FWCore.ParameterSet.Config as cms

btagMC_QCD_380_470 = cms.EDFilter("BTagSkimMC",
    mcProcess = cms.untracked.string('QCD'),
    pthat_min = cms.untracked.double(380.0),
    verbose = cms.untracked.bool(False),
    pthat_max = cms.untracked.double(470.0)
)


