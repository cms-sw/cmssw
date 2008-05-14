import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelVertexFinding.PixelVertexes_cfi import *
from FastSimulation.Tracking.PixelTracksProducer_cff import *
pixelVertexing = cms.Sequence(pixelVertices)
recopixelvertexing = cms.Sequence(pixelTracking+pixelVertices)
# A copy of the above
HLTRecopixelvertexingSequence = cms.Sequence(hltPixelTracking+cms.SequencePlaceholder("hltPixelVertices"))

