
import FWCore.ParameterSet.Config as cms

trackingFailureFilter = cms.EDFilter(
  "TrackingFailureFilter",
  JetSource = cms.InputTag('ak4PFJets'),
  TrackSource = cms.InputTag('generalTracks'),
  VertexSource = cms.InputTag('goodVertices'),
  DzTrVtxMax = cms.double(1),
  DxyTrVtxMax = cms.double(0.2),
  MinSumPtOverHT = cms.double(0.10),
  taggingMode = cms.bool(False),
  debug = cms.bool(False),
)
