import FWCore.ParameterSet.Config as cms

from FastSimulation.Tracking.PixelTracksProducer_cff import *
HLTRecopixelvertexingSequence = cms.Sequence(hltPixelTracking+cms.SequencePlaceholder("hltPixelVertices"))
