import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.secondaryVertexTagInfos_cfi import *

inclusiveSecondaryVertexFinderTagInfos = secondaryVertexTagInfos.clone()

# use external SV collection made from IVF
inclusiveSecondaryVertexFinderTagInfos.extSVCollection     = cms.InputTag('inclusiveMergedVertices')
inclusiveSecondaryVertexFinderTagInfos.extSVDeltaRToJet    = cms.double(0.3)
inclusiveSecondaryVertexFinderTagInfos.useExternalSV = cms.bool(True)
inclusiveSecondaryVertexFinderTagInfos.vertexCuts.fracPV = 0.79 ## 4 out of 5 is discarded
inclusiveSecondaryVertexFinderTagInfos.vertexCuts.distSig2dMin = 2.0



# filtered IVF as used in some analyses

inclusiveSecondaryVertexFinderFilteredTagInfos = secondaryVertexTagInfos.clone()

inclusiveSecondaryVertexFinderFilteredTagInfos.extSVCollection     = cms.InputTag('bToCharmDecayVertexMerged')
inclusiveSecondaryVertexFinderFilteredTagInfos.extSVDeltaRToJet    = cms.double(0.5)
inclusiveSecondaryVertexFinderFilteredTagInfos.useExternalSV = cms.bool(True)
inclusiveSecondaryVertexFinderFilteredTagInfos.vertexCuts.fracPV = 0.79 ## 4 out of 5 is discarded
inclusiveSecondaryVertexFinderFilteredTagInfos.vertexCuts.distSig2dMin = 2.0

