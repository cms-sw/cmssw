import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.trackSelection_cfi import *
from RecoBTag.SecondaryVertex.trackPseudoSelection_cfi import *

combinedSecondaryVertexCommon = cms.PSet(
	trackPseudoSelectionBlock,
	trackSelectionBlock,
	trackFlip = cms.bool(False),
	vertexFlip = cms.bool(False),
	useTrackWeights = cms.bool(True),
	pseudoMultiplicityMin = cms.uint32(2),
	correctVertexMass = cms.bool(True),
	trackPairV0Filter = cms.PSet(k0sMassWindow = cms.double(0.03)),
	charmCut = cms.double(1.5),
	minimumTrackWeight = cms.double(0.5),
	pseudoVertexV0Filter = cms.PSet(k0sMassWindow = cms.double(0.05)),
	trackMultiplicityMin = cms.uint32(2),
	trackSort = cms.string('sip2dSig')
)
