import FWCore.ParameterSet.Config as cms

from RecoBTag.DeepFlavour.pfDeepFlavourJetTags_cfi import pfDeepFlavourJetTags

pfNegativeDeepFlavourJetTags = pfDeepFlavourJetTags.clone(
        src = cms.InputTag('pfNegativeDeepFlavourTagInfos')
        )
