import FWCore.ParameterSet.Config as cms

hltVertexFilter = cms.EDFilter('HLTVertexFilter',
  inputTag    = cms.InputTag( 'hltPixelVertices' ),
  saveTags = cms.bool( False ),
  minVertices = cms.uint32( 1 ),
  minNDoF     = cms.double( 0. ),
  maxChi2     = cms.double( 99999. ),
  maxD0       = cms.double( 1. ),
  maxZ        = cms.double( 15. )
)
