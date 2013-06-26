import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.trackSelection_cfi import *

ghostTrackCommon = cms.PSet(
	trackSelectionBlock,
	trackPairV0Filter = cms.PSet(k0sMassWindow = cms.double(0.03)),
	charmCut = cms.double(1.5),
	minimumTrackWeight = cms.double(0.5),
	trackSort = cms.string('sip2dSig')
)
