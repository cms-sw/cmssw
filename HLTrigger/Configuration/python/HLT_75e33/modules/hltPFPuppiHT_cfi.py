import FWCore.ParameterSet.Config as cms

hltPFPuppiHT = cms.EDProducer("HLTHtMhtProducer",
    excludePFMuons = cms.bool(False),
    jetsLabel = cms.InputTag("hltAK4PFPuppiJetsCorrected"),
    maxEtaJetHt = cms.double(2.4),
    maxEtaJetMht = cms.double(2.4),
    minNJetHt = cms.int32(0),
    minNJetMht = cms.int32(0),
    minPtJetHt = cms.double(30.0),
    minPtJetMht = cms.double(30.0),
    pfCandidatesLabel = cms.InputTag(""),
    usePt = cms.bool(True)
)
