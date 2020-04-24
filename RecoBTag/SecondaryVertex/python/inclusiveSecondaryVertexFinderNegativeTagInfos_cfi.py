import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.inclusiveSecondaryVertexFinderTagInfos_cfi import *

inclusiveSecondaryVertexFinderNegativeTagInfos = inclusiveSecondaryVertexFinderTagInfos.clone()
inclusiveSecondaryVertexFinderNegativeTagInfos.extSVDeltaRToJet = cms.double(-0.3)
inclusiveSecondaryVertexFinderNegativeTagInfos.vertexCuts.distVal2dMin = -2.5
inclusiveSecondaryVertexFinderNegativeTagInfos.vertexCuts.distVal2dMax = -0.01
inclusiveSecondaryVertexFinderNegativeTagInfos.vertexCuts.distSig2dMin = -99999.9
inclusiveSecondaryVertexFinderNegativeTagInfos.vertexCuts.distSig2dMax = -2.0
inclusiveSecondaryVertexFinderNegativeTagInfos.vertexCuts.maxDeltaRToJetAxis = -0.5
