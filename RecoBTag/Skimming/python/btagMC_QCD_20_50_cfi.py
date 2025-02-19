import FWCore.ParameterSet.Config as cms

btagMC_QCD_20_50 = cms.EDFilter("BTagSkimMC",
    mcProcess = cms.string('QCD'),
    pthat_min = cms.double(20.0),
    verbose = cms.untracked.bool(False),
    pthat_max = cms.double(50.0)
)


