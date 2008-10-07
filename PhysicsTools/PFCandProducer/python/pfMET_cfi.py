import FWCore.ParameterSet.Config as cms


pfMET = cms.EDProducer("PFMET",
    PFCandidates = cms.InputTag("particleFlow"),
    verbose = cms.untracked.bool(False)
)

