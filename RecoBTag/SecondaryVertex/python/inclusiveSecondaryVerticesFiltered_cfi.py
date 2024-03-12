import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.bVertexFilter_cfi import *

inclusiveSecondaryVerticesFiltered = bVertexFilter.clone(
    vertexFilter = dict(multiplicityMin = 2),
    secondaryVertices = "inclusiveSecondaryVertices"
)
# foo bar baz
# seSLAl5KyXXWR
# oR19CjLjHsEtO
