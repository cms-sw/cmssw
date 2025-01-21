import FWCore.ParameterSet.Config as cms

from ..modules.hltAK4PFJetsForTaus_cfi import *
from ..modules.hltHpsDoublePFTau35MediumDitauWPDeepTau_cfi import *
from ..modules.hltHpsSelectedPFTausMediumDitauWPDeepTau_cfi import *
from ..modules.hltPreDoublePFTauHPS_cfi import *
from ..sequences.HLTHgcalLocalRecoSequence_cfi import *
from ..sequences.HLTAK4PFJetsReconstruction_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.HLTHPSDeepTauPFTauSequence_cfi import *
from ..sequences.HLTMuonsSequence_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *
from ..sequences.HLTPFTauHPS_cfi import *
from ..sequences.HLTTrackingSequence_cfi import *
from ..sequences.HLTLocalrecoSequence_cfi import *
from ..sequences.HLTRawToDigiSequence_cfi import *
from ..modules.hltL1SeedForDoublePuppiTau_cfi import *

HLT_DoubleMediumDeepTauPFTauHPS35_eta2p1 = cms.Path(
    HLTBeginSequence
    + hltL1SeedForDoublePuppiTau
    + hltPreDoublePFTauHPS
    + HLTRawToDigiSequence
    + HLTHgcalLocalRecoSequence
    + HLTLocalrecoSequence
    + HLTTrackingSequence
    + HLTMuonsSequence
    + HLTParticleFlowSequence
    + HLTAK4PFJetsReconstruction
    + hltAK4PFJetsForTaus
    + HLTPFTauHPS
    + HLTHPSDeepTauPFTauSequence
    + hltHpsSelectedPFTausMediumDitauWPDeepTau
    + hltHpsDoublePFTau35MediumDitauWPDeepTau
    + HLTEndSequence
)
