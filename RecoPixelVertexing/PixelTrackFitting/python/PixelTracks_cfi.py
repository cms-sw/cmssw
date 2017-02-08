import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelTrackFitting.PixelTrackReconstruction_cfi import *

pixelTracks = cms.EDProducer("PixelTrackProducer",
  PixelTrackReconstructionBlock, 
  passLabel  = cms.string('pixelTracks')
)


