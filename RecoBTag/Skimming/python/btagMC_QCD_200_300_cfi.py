import FWCore.ParameterSet.Config as cms

btagMC_QCD_200_300 = cms.EDFilter("BTagSkimMC",
    mcProcess = cms.untracked.string('QCD'),
    pthat_min = cms.untracked.double(200.0),
    verbose = cms.untracked.bool(False),
    pthat_max = cms.untracked.double(300.0)
)


