import FWCore.ParameterSet.Config as cms

btagMC_QCD_800_1000 = cms.EDFilter("BTagSkimMC",
    mcProcess = cms.string('QCD'),
    pthat_min = cms.double(800.0),
    verbose = cms.untracked.bool(False),
    pthat_max = cms.double(1000.0)
)


