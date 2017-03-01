import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.bVertexFilter_cfi import *

inclusiveSecondaryVerticesFiltered = bVertexFilter.clone()
inclusiveSecondaryVerticesFiltered.vertexFilter.multiplicityMin = 2
inclusiveSecondaryVerticesFiltered.secondaryVertices = cms.InputTag("inclusiveSecondaryVertices")
