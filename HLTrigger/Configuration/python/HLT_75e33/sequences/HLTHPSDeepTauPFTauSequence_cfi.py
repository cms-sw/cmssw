import FWCore.ParameterSet.Config as cms

from ..modules.hltHpsPFTauDiscriminationByDecayModeFindingNewDMs_cfi import *
from ..modules.hltHpsPFTauPrimaryVertexProducerForDeepTau_cfi import *
from ..modules.hltHpsPFTauSecondaryVertexProducerForDeepTau_cfi import *
from ..modules.hltHpsPFTauTransverseImpactParametersForDeepTau_cfi import *
from ..modules.hltFixedGridRhoProducerFastjetAllTau_cfi import *
from ..modules.hltHpsPFTauBasicDiscriminatorsForDeepTau_cfi import *
from ..modules.hltHpsPFTauBasicDiscriminatorsdR03ForDeepTau_cfi import *
from ..modules.hltHpsPFTauDeepTauProducer_cfi import *

HLTHPSDeepTauPFTauSequence = cms.Sequence( 
    hltHpsPFTauDiscriminationByDecayModeFindingNewDMs +
    hltHpsPFTauPrimaryVertexProducerForDeepTau +
    hltHpsPFTauSecondaryVertexProducerForDeepTau +
    hltHpsPFTauTransverseImpactParametersForDeepTau +
    hltFixedGridRhoProducerFastjetAllTau + 
    hltHpsPFTauBasicDiscriminatorsForDeepTau +
    hltHpsPFTauBasicDiscriminatorsdR03ForDeepTau + 
    hltHpsPFTauDeepTauProducer 
)
