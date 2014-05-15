import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelVertexFinding.PVClusterComparer_cfi import *

pixelVertexCollectionTrimmer = cms.EDProducer('PixelVertexCollectionTrimmer',
  src            = cms.InputTag(''),
  maxVtx         = cms.uint32(100) ,
  fractionSumPt2 = cms.double(0.3) ,
  minSumPt2      = cms.double(0.),
  PVcomparer = cms.PSet(
     refToPSet_ = cms.string('pvClusterComparer')
  )
)
