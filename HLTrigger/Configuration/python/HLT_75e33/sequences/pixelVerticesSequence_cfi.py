import FWCore.ParameterSet.Config as cms

from ..modules.pixelVertices_cfi import *

pixelVerticesSequence = cms.Sequence(pixelVertices)
