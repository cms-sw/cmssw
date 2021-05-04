import FWCore.ParameterSet.Config as cms

l1tPFPuppiHTMaxEta2p4 = cms.EDProducer("HLTHtMhtProducer",
    jetsLabel = cms.InputTag("l1tSlwPFPuppiJetsCorrected","Phase1L1TJetFromPfCandidates"),
    maxEtaJetHt = cms.double(2.4),
    minPtJetHt = cms.double(30.0)
)
