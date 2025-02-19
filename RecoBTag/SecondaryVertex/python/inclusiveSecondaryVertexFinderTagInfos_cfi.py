import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.secondaryVertexTagInfos_cfi import *

inclusiveSecondaryVertexFinderTagInfos = secondaryVertexTagInfos.clone()

# use external SV collection made from IVF
inclusiveSecondaryVertexFinderTagInfos.extSVCollection     = cms.InputTag('inclusiveMergedVertices')
inclusiveSecondaryVertexFinderTagInfos.extSVDeltaRToJet    = cms.double(0.3)
inclusiveSecondaryVertexFinderTagInfos.useExternalSV = cms.bool(True)
inclusiveSecondaryVertexFinderTagInfos.vertexCuts.fracPV = 0.79 ## 4 out of 5 is discarded
inclusiveSecondaryVertexFinderTagInfos.vertexCuts.distSig2dMin = 2.0

