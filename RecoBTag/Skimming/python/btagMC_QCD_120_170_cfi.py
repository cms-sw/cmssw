import FWCore.ParameterSet.Config as cms

btagMC_QCD_120_170 = cms.EDFilter("BTagSkimMC",
    mcProcess = cms.string('QCD'),
    pthat_min = cms.double(120.0),
    verbose = cms.untracked.bool(False),
    pthat_max = cms.double(170.0)
)


