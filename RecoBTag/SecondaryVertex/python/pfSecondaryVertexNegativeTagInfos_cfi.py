import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.pfSecondaryVertexTagInfos_cfi import *

pfSecondaryVertexNegativeTagInfos = pfSecondaryVertexTagInfos.clone()
pfSecondaryVertexNegativeTagInfos.vertexCuts.distVal2dMin = -2.5
pfSecondaryVertexNegativeTagInfos.vertexCuts.distVal2dMax = -0.01
pfSecondaryVertexNegativeTagInfos.vertexCuts.distSig2dMin = -99999.9
pfSecondaryVertexNegativeTagInfos.vertexCuts.distSig2dMax = -3.0
pfSecondaryVertexNegativeTagInfos.vertexCuts.maxDeltaRToJetAxis = -0.5
