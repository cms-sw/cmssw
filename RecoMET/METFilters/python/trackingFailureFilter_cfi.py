
import FWCore.ParameterSet.Config as cms

trackingFailureFilter = cms.EDFilter(
  "TrackingFailureFilter",
  JetSource = cms.InputTag('patJetsAK5PF'),
  TrackSource = cms.InputTag('generalTracks'),
  VertexSource = cms.InputTag('goodVerticesRA2'),
  DzTrVtxMax = cms.double(1),
  DxyTrVtxMax = cms.double(0.2),
  MinSumPtOverHT = cms.double(0.10)
)
