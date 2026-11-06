import FWCore.ParameterSet.Config as cms

l1tTrackerHTMiss = cms.EDProducer("L1TkHTMissProducer",
    L1TkJetInputTag = cms.InputTag("l1tTrackJets","L1TrackJets"),
    L1VertexInputTag = cms.InputTag("l1tVertexFinder","L1Vertices"),
    deltaZ = cms.double(1.0),
    displaced = cms.bool(False),
    doVtxConstrain = cms.bool(False),
    jet_maxEta = cms.double(2.4),
    jet_minJetEtHighPt = cms.double(100.0),
    jet_minJetEtLowPt = cms.double(50.0),
    jet_minNtracksHighPt = cms.int32(0),
    jet_minNtracksLowPt = cms.int32(0),
    jet_minPt = cms.double(3.0),
    primaryVtxConstrain = cms.bool(False),
    useCaloJets = cms.bool(False)
)
