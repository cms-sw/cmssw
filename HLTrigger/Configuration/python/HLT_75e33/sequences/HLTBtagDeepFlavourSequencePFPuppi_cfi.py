import FWCore.ParameterSet.Config as cms

from ..modules.hltDeepBLifetimeTagInfosPFPuppi_cfi import *
from ..modules.hltDeepCombinedSecondaryVertexBJetTagsInfosPuppi_cfi import *
from ..modules.hltDeepInclusiveMergedVerticesPF_cfi import *
from ..modules.hltDeepInclusiveSecondaryVerticesPF_cfi import *
from ..modules.hltDeepInclusiveVertexFinderPF_cfi import *
from ..modules.hltDeepSecondaryVertexTagInfosPFPuppi_cfi import *
from ..modules.hltDeepTrackVertexArbitratorPF_cfi import *
from ..modules.hltPfDeepFlavourJetTags_cfi import *
from ..modules.hltPfDeepFlavourTagInfos_cfi import *
from ..modules.hltPrimaryVertexAssociation_cfi import *

HLTBtagDeepFlavourSequencePFPuppi = cms.Sequence(
    hltDeepBLifetimeTagInfosPFPuppi +
    hltDeepInclusiveVertexFinderPF +
    hltDeepInclusiveSecondaryVerticesPF +
    hltDeepTrackVertexArbitratorPF +
    hltDeepInclusiveMergedVerticesPF +
    hltDeepSecondaryVertexTagInfosPFPuppi +
    hltPrimaryVertexAssociation +
    hltDeepCombinedSecondaryVertexBJetTagsInfosPuppi +
    hltPfDeepFlavourTagInfos +
    hltPfDeepFlavourJetTags
)
