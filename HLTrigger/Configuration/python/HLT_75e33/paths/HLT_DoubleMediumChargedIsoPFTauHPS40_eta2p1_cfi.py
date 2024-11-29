import FWCore.ParameterSet.Config as cms

from ..modules.hltAK4PFJetsForTaus_cfi import *
from ..modules.hltHpsDoublePFTau40TrackPt1MediumChargedIsolation_cfi import *
from ..modules.hltHpsSelectedPFTausTrackPt1MediumChargedIsolation_cfi import *
from ..modules.hltPreDoublePFTauHPS_cfi import *
from ..sequences.HLTHgcalLocalRecoSequence_cfi import *
from ..sequences.HLTAK4PFJetsReconstruction_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.HLTHPSMediumChargedIsoPFTauSequence_cfi import *
from ..sequences.HLTMuonsSequence_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *
from ..sequences.HLTPFTauHPS_cfi import *
from ..sequences.HLTTrackingV61Sequence_cfi import *
from ..sequences.HLTLocalrecoSequence_cfi import *
from ..sequences.HLTRawToDigiSequence_cfi import *
from ..modules.hltL1SeedForDoublePuppiTau_cfi import *

HLT_DoubleMediumChargedIsoPFTauHPS40_eta2p1 = cms.Path(
    HLTBeginSequence
    + hltL1SeedForDoublePuppiTau
    + hltPreDoublePFTauHPS
    + HLTRawToDigiSequence
    + HLTHgcalLocalRecoSequence
    + HLTLocalrecoSequence
    + HLTTrackingV61Sequence
    + HLTMuonsSequence
    + HLTParticleFlowSequence
    + HLTAK4PFJetsReconstruction
    + hltAK4PFJetsForTaus
    + HLTPFTauHPS
    + HLTHPSMediumChargedIsoPFTauSequence
    + hltHpsSelectedPFTausTrackPt1MediumChargedIsolation
    + hltHpsDoublePFTau40TrackPt1MediumChargedIsolation
    + HLTEndSequence
)
