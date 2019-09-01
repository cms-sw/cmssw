import FWCore.ParameterSet.Config as cms

from RecoBTag.FeatureTools.pfDeepFlavourTagInfos_cfi import pfDeepFlavourTagInfos

pfNegativeDeepFlavourTagInfos = pfDeepFlavourTagInfos.clone(
    shallow_tag_infos = 'pfDeepCSVNegativeTagInfos',
    secondary_vertices = 'inclusiveCandidateNegativeSecondaryVertices',
    flip = True
    )
