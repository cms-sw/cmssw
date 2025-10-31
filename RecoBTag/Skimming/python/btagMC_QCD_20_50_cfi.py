import FWCore.ParameterSet.Config as cms

btagMC_QCD_20_50 = cms.EDFilter("BTagSkimMC",
    mcProcess = cms.untracked.string('QCD'),
    pthat_min = cms.untracked.double(20.0),
    verbose = cms.untracked.bool(False),
    pthat_max = cms.untracked.double(50.0)
)


