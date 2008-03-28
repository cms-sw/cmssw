import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelTrackFitting.PixelTracks_cff import *
from RecoPixelVertexing.PixelVertexFinding.PixelVertexes_cff import *
recopixelvertexing = cms.Sequence(pixelTracks*pixelVertices)

