import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.Configuration.RecoPixelVertexing_cff import *
hltBLifetimeL25tracking = cms.Sequence(cms.SequencePlaceholder("doLocalPixel")+recopixelvertexing)

