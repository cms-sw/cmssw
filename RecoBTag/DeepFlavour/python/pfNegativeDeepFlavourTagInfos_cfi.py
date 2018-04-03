import FWCore.ParameterSet.Config as cms

from RecoBTag.DeepFlavour.pfDeepFlavourTagInfos_cfi import pfDeepFlavourTagInfos

pfNegativeDeepFlavourTagInfos = pfDeepFlavourTagInfos.clone(
        shallow_tag_infos = cms.InputTag('pfDeepCSVNegativeTagInfos'),
        flip = cms.bool(True)
        )
