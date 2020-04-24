import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.secondaryVertexTagInfos_cfi import *

inclusiveSecondaryVertexFinderFilteredTagInfos = secondaryVertexTagInfos.clone()

# filtered IVF as used in some analyses
inclusiveSecondaryVertexFinderFilteredTagInfos.extSVCollection     = cms.InputTag('bToCharmDecayVertexMerged')
inclusiveSecondaryVertexFinderFilteredTagInfos.extSVDeltaRToJet    = cms.double(0.4)
inclusiveSecondaryVertexFinderFilteredTagInfos.useExternalSV = cms.bool(True)
inclusiveSecondaryVertexFinderFilteredTagInfos.vertexCuts.fracPV = 0.79 ## 4 out of 5 is discarded
inclusiveSecondaryVertexFinderFilteredTagInfos.vertexCuts.distSig2dMin = 2.0

