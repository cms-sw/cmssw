import copy
import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.secondaryVertexTagInfos_cfi import *

secondaryVertexNegativeTagInfos = copy.deepcopy(secondaryVertexTagInfos)
secondaryVertexNegativeTagInfos.vertexCuts.distVal2dMin = -2.5
secondaryVertexNegativeTagInfos.vertexCuts.distVal2dMax = -0.01
secondaryVertexNegativeTagInfos.vertexCuts.distSig2dMin = -99999.9
secondaryVertexNegativeTagInfos.vertexCuts.distSig2dMax = -3.0
secondaryVertexNegativeTagInfos.vertexCuts.maxDeltaRToJetAxis = -0.5
