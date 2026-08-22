import FWCore.ParameterSet.Config as cms

from ..modules.hltDeepInclusiveMergedVerticesPF_cfi import *
from ..modules.hltDeepInclusiveSecondaryVerticesPF_cfi import *
from ..modules.hltDeepInclusiveVertexFinderPF_cfi import *
from ..modules.hltPrimaryVertexAssociation_cfi import *
from ..modules.hltDeepTrackVertexArbitratorPF_cfi import *
from ..modules.hltPFJetForBtagSelector_cfi import *
from ..modules.hltPFJetForBtag_cfi import *
from ..modules.hltParticleTransformerAK4TagInfos_cfi import *
from ..modules.hltParticleTransformerONNXJetTags_cfi import *
from ..modules.hltParticleTransformerDiscriminatorsJetTags_cfi import *

HLTJetFlavourTagParticleTransformerSequencePF = cms.Sequence( 
    hltPFJetForBtagSelector + 
    hltPFJetForBtag + 
    hltDeepInclusiveVertexFinderPF + 
    hltDeepInclusiveSecondaryVerticesPF + 
    hltDeepTrackVertexArbitratorPF + 
    hltDeepInclusiveMergedVerticesPF + 
    hltPrimaryVertexAssociation + 
    hltParticleTransformerAK4TagInfos + 
    hltParticleTransformerONNXJetTags + 
    hltParticleTransformerDiscriminatorsJetTags 
    )
