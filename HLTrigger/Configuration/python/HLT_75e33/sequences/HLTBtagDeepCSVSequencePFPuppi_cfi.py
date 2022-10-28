import FWCore.ParameterSet.Config as cms

from ..modules.hltDeepBLifetimeTagInfosPFPuppi_cfi import *
from ..modules.hltDeepCombinedSecondaryVertexBJetTagsInfosPuppi_cfi import *
from ..modules.hltDeepCombinedSecondaryVertexBJetTagsPFPuppi_cfi import *
from ..modules.hltDeepInclusiveMergedVerticesPF_cfi import *
from ..modules.hltDeepInclusiveSecondaryVerticesPF_cfi import *
from ..modules.hltDeepInclusiveVertexFinderPF_cfi import *
from ..modules.hltDeepSecondaryVertexTagInfosPFPuppi_cfi import *
from ..modules.hltDeepTrackVertexArbitratorPF_cfi import *

HLTBtagDeepCSVSequencePFPuppi = cms.Sequence(
    hltDeepBLifetimeTagInfosPFPuppi +
    hltDeepInclusiveVertexFinderPF +
    hltDeepInclusiveSecondaryVerticesPF +
    hltDeepTrackVertexArbitratorPF +
    hltDeepInclusiveMergedVerticesPF +
    hltDeepSecondaryVertexTagInfosPFPuppi +
    hltDeepCombinedSecondaryVertexBJetTagsInfosPuppi +
    hltDeepCombinedSecondaryVertexBJetTagsPFPuppi
)
