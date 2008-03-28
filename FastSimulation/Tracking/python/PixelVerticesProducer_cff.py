import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelVertexFinding.PixelVertexes_cfi import *
from FastSimulation.Tracking.PixelTracksProducer_cff import *
pixelGSVertexing = cms.Sequence(pixelVertices)
recopixelvertexing = cms.Sequence(pixelGSTracking+pixelVertices)

