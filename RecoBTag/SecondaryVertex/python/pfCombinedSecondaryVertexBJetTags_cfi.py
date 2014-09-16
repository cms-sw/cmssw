import FWCore.ParameterSet.Config as cms
from RecoBTag.SecondaryVertex.combinedSecondaryVertexBJetTags_cfi import *
pfCombinedSecondaryVertexBJetTags = combinedSecondaryVertexBJetTags.clone(
    jetTagComputer = cms.string('candidateCombinedSecondaryVertex'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"), cms.InputTag("pfSecondaryVertexTagInfos"))
)

#add V1 and V2
