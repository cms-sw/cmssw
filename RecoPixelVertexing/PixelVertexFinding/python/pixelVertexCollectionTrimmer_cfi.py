import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelVertexFinding.PVClusterComparer_cfi import *

pixelVertexCollectionTrimmer = cms.EDProducer('PixelVertexCollectionTrimmer',
  src            = cms.InputTag(''),
  maxVtx         = cms.int32(100) ,
  fractionSumPt2 = cms.double(0.3) ,
  minSumPt2      = cms.double(0.),
  PVcomparer = cms.PSet(
     refToPSet_ = cms.string('pvClusterComparer')
  )
#  PVcomparer = cms.PSet(
#      track_pt_min   = cms.double(     1.0),
#      track_pt_max   = cms.double(    10.0), # SD: 20.
#      track_chi2_max = cms.double(999999. ), # SD: 20
#      track_prob_min = cms.double(    -1. ), # RM: 0.001
#   )
)
