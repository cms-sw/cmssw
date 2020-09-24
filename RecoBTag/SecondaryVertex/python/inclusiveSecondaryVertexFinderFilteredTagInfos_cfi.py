import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.secondaryVertexTagInfos_cfi import *

inclusiveSecondaryVertexFinderFilteredTagInfos = secondaryVertexTagInfos.clone(
# filtered IVF as used in some analyses
    extSVCollection  = 'bToCharmDecayVertexMerged',
    extSVDeltaRToJet = 0.4,
    useExternalSV    = True,
    vertexCuts = dict(fracPV = 0.79, ## 4 out of 5 is discarded
                      distSig2dMin = 2.0)
)
