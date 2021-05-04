import FWCore.ParameterSet.Config as cms

from ..modules.hltDeepBLifetimeTagInfosPFPuppiModEta2p4_cfi import *
from ..modules.hltDeepCombinedSecondaryVertexBJetTagsInfosPuppiModEta2p4_cfi import *
from ..modules.hltDeepInclusiveMergedVerticesPF_cfi import *
from ..modules.hltDeepInclusiveSecondaryVerticesPF_cfi import *
from ..modules.hltDeepInclusiveVertexFinderPF_cfi import *
from ..modules.hltDeepSecondaryVertexTagInfosPFPuppiModEta2p4_cfi import *
from ..modules.hltDeepTrackVertexArbitratorPF_cfi import *
from ..modules.hltPfDeepFlavourJetTagsModEta2p4_cfi import *
from ..modules.hltPfDeepFlavourTagInfosModEta2p4_cfi import *
from ..modules.hltPFPuppiJetForBtagEta2p4_cfi import *
from ..modules.hltPFPuppiJetForBtagSelectorEta2p4_cfi import *
from ..modules.hltPrimaryVertexAssociationModEta2p4_cfi import *

HLTBtagDeepFlavourSequencePFPuppiModEta2p4 = cms.Sequence(
    hltPFPuppiJetForBtagSelectorEta2p4 +
    hltPFPuppiJetForBtagEta2p4 +
    hltDeepBLifetimeTagInfosPFPuppiModEta2p4 +
    hltDeepInclusiveVertexFinderPF +
    hltDeepInclusiveSecondaryVerticesPF +
    hltDeepTrackVertexArbitratorPF +
    hltDeepInclusiveMergedVerticesPF +
    hltDeepSecondaryVertexTagInfosPFPuppiModEta2p4 +
    hltPrimaryVertexAssociationModEta2p4 +
    hltDeepCombinedSecondaryVertexBJetTagsInfosPuppiModEta2p4 +
    hltPfDeepFlavourTagInfosModEta2p4 +
    hltPfDeepFlavourJetTagsModEta2p4
)
