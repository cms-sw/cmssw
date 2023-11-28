import FWCore.ParameterSet.Config as cms

from ..modules.hltTauPFJets08Region_cfi import *
from ..modules.hltHpsTauPFJetsRecoTauChargedHadronsWithNeutrals_cfi import *
from ..modules.hltPFTauPiZeros_cfi import *
from ..modules.hltHpsCombinatoricRecoTaus_cfi import *
from ..modules.hltHpsSelectionDiscriminator_cfi import *
from ..modules.hltHpsPFTauProducerSansRefs_cfi import *
from ..modules.hltHpsPFTauProducer_cfi import *
from ..modules.hltHpsPFTauDiscriminationByDecayModeFindingNewDMs_cfi import *
from ..modules.hltHpsPFTauTrackFindingDiscriminator_cfi import *
from ..modules.hltHpsSelectedPFTausTrackFinding_cfi import *
from ..modules.hltHpsPFTauTrack_cfi import *

HLTPFTauHPS = cms.Sequence(
    hltTauPFJets08Region +
    hltHpsTauPFJetsRecoTauChargedHadronsWithNeutrals + 
    hltPFTauPiZeros + 
    hltHpsCombinatoricRecoTaus + 
    hltHpsSelectionDiscriminator + 
    hltHpsPFTauProducerSansRefs + 
    hltHpsPFTauProducer + 
    hltHpsPFTauDiscriminationByDecayModeFindingNewDMs + 
    hltHpsPFTauTrackFindingDiscriminator + 
    hltHpsSelectedPFTausTrackFinding + 
    hltHpsPFTauTrack 
)
