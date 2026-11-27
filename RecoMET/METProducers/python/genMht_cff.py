import FWCore.ParameterSet.Config as cms

genMht = cms.EDProducer(
    "HLTHtMhtProducer",
    jetsLabel=cms.InputTag("ak4GenJetsNoNu"),
    maxEtaJetHt=cms.double(5.5),
    maxEtaJetMht=cms.double(5.5),
    minNJetHt=cms.int32(0),
    minNJetMht=cms.int32(0),
    minPtJetHt=cms.double(20.0),
    minPtJetMht=cms.double(20.0),
    excludePFMuons=cms.bool(False),
    pfCandidatesLabel=cms.InputTag(""),
    usePt=cms.bool(False),
)
