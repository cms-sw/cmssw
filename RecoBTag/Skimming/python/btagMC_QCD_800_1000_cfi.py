import FWCore.ParameterSet.Config as cms

btagMC_QCD_800_1000 = cms.EDFilter("BTagSkimMC",
    mcProcess = cms.untracked.string('QCD'),
    pthat_min = cms.untracked.double(800.0),
    verbose = cms.untracked.bool(False),
    pthat_max = cms.untracked.double(1000.0)
)


