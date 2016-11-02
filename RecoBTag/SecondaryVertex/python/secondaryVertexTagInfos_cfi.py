import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.vertexTrackSelection_cff import *
from RecoBTag.SecondaryVertex.vertexReco_cff import *
from RecoBTag.SecondaryVertex.vertexCuts_cff import *
from RecoBTag.SecondaryVertex.vertexSelection_cff import *

secondaryVertexTagInfos = cms.EDProducer("SecondaryVertexProducer",
	vertexTrackSelectionBlock,
	vertexSelectionBlock,
	vertexCutsBlock,
	vertexRecoBlock,
	constraint = cms.string("BeamSpot"),
	trackIPTagInfos = cms.InputTag("impactParameterTagInfos"),
	minimumTrackWeight = cms.double(0.5),
	usePVError = cms.bool(True),
	trackSort = cms.string('sip3dSig'),
        beamSpotTag = cms.InputTag('offlineBeamSpot'),                                        
        useExternalSV       = cms.bool(False),
        extSVCollection     = cms.InputTag('secondaryVertices'),
        extSVDeltaRToJet    = cms.double(0.3)

)

secondaryVertexTagInfos.trackSelection.pixelHitsMin = cms.uint32(2)
secondaryVertexTagInfos.trackSelection.totalHitsMin = cms.uint32(8)
