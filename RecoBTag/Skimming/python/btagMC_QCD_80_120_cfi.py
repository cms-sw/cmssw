import FWCore.ParameterSet.Config as cms

btagMC_QCD_80_120 = cms.EDFilter("BTagSkimMC",
    mcProcess = cms.untracked.string('QCD'),
    pthat_min = cms.untracked.double(80.0),
    verbose = cms.untracked.bool(False),
    pthat_max = cms.untracked.double(120.0)
)


