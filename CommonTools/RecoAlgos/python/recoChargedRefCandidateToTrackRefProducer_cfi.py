import FWCore.ParameterSet.Config as cms

recoChargedRefCandidateToTrackRefProducer = cms.EDProducer("RecoChargedRefCandidateToTrackRefProducer",
    src = cms.InputTag('trackRefsForJets')
)
# foo bar baz
# 8ElQeMLRxLuaj
