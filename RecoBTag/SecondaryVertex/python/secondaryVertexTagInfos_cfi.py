import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.trackSelection_cfi import *
from RecoBTag.SecondaryVertex.vertexReco_cfi import *
from RecoBTag.SecondaryVertex.vertexCuts_cfi import *
from RecoBTag.SecondaryVertex.vertexSelection_cfi import *
secondaryVertexTagInfos = cms.EDProducer("SecondaryVertexProducer",
    trackSelectionBlock,
    vertexSelectionBlock,
    vertexCutsBlock,
    vertexRecoBlock,
    useBeamConstraint = cms.bool(True),
    trackIPTagInfos = cms.InputTag("impactParameterTagInfos"),
    minimumTrackWeight = cms.double(0.5),
    usePVError = cms.bool(True),
    trackSort = cms.string('sip3dSig')
)


