import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.trackSelection_cff import *

pfBoostedDoubleSVAK8TagInfos = cms.EDProducer("BoostedDoubleSVProducer",
    trackSelectionBlock,
    beta = cms.double(1.0),
    R0 = cms.double(0.8),
    maxSVDeltaRToJet = cms.double(0.7),
    trackPairV0Filter = cms.PSet(k0sMassWindow = cms.double(0.03)),
    svTagInfos = cms.InputTag("pfInclusiveSecondaryVertexFinderAK8TagInfos")
)

pfBoostedDoubleSVAK8TagInfos.trackSelection.jetDeltaRMax = cms.double(0.8)
