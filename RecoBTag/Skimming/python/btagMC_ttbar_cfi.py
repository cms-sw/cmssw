import FWCore.ParameterSet.Config as cms

btagMC_ttbar = cms.EDFilter("BTagSkimMC",
    mcProcess = cms.untracked.string('ttbar'),
    pthat_min = cms.untracked.double(50.0),
    verbose = cms.untracked.bool(False),
    pthat_max = cms.untracked.double(80.0)
)


