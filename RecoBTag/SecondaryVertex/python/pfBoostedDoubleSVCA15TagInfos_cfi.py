import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.trackSelection_cff import *

pfBoostedDoubleSVCA15TagInfos = cms.EDProducer("BoostedDoubleSVProducer",
    trackSelectionBlock,
    beta = cms.double(1.0),
    R0 = cms.double(1.5),
    maxSVDeltaRToJet = cms.double(1.0),
    trackPairV0Filter = cms.PSet(k0sMassWindow = cms.double(0.03)),
    svTagInfos = cms.InputTag("pfInclusiveSecondaryVertexFinderCA15TagInfos")
)

pfBoostedDoubleSVCA15TagInfos.trackSelection.jetDeltaRMax = cms.double(1.5)
