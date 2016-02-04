import FWCore.ParameterSet.Config as cms

btagMC_QCD_200_300 = cms.EDFilter("BTagSkimMC",
    mcProcess = cms.string('QCD'),
    pthat_min = cms.double(200.0),
    verbose = cms.untracked.bool(False),
    pthat_max = cms.double(300.0)
)


