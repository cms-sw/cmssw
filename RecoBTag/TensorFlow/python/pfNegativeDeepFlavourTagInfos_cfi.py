import FWCore.ParameterSet.Config as cms

from RecoBTag.TensorFlow.pfDeepFlavourTagInfos_cfi import pfDeepFlavourTagInfos

pfNegativeDeepFlavourTagInfos = pfDeepFlavourTagInfos.clone(
    shallow_tag_infos = cms.InputTag('pfDeepCSVNegativeTagInfos'),
    secondary_vertices = cms.InputTag('inclusiveCandidateNegativeSecondaryVertices'),
    flip = cms.bool(True)
    )
