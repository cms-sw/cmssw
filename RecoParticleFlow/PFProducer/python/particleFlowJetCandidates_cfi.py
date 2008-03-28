import FWCore.ParameterSet.Config as cms

particleFlowJetCandidates = cms.EDFilter("PFJetCandidateCreator",
    src = cms.InputTag("particleFlow"),
    verbose = cms.untracked.bool(True)
)


