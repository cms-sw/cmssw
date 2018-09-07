import FWCore.ParameterSet.Config as cms

from RecoBTag.TensorFlow.pfDeepFlavourJetTags_cfi import pfDeepFlavourJetTags

pfNegativeDeepFlavourJetTags = pfDeepFlavourJetTags.clone(
        src = cms.InputTag('pfNegativeDeepFlavourTagInfos')
        )
