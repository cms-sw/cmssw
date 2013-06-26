import FWCore.ParameterSet.Config as cms

btagMC_ttbar = cms.EDFilter("BTagSkimMC",
    mcProcess = cms.string('ttbar'),
    pthat_min = cms.double(50.0),
    verbose = cms.untracked.bool(False),
    pthat_max = cms.double(80.0)
)


