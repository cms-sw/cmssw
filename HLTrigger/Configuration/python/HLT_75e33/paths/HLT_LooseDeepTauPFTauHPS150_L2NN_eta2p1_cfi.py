import FWCore.ParameterSet.Config as cms

from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTRawToDigiSequence_cfi import *
from ..sequences.HLTTICLLocalRecoSequence_cfi import *
from ..sequences.HLTLocalrecoSequence_cfi import *
from ..sequences.HLTTrackingSequence_cfi import *
from ..sequences.HLTMuonsSequence_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *
from ..sequences.HLTAK4PFJetsReconstruction_cfi import *
from ..sequences.HLTPFTauHPS_cfi import *
from ..sequences.HLTHPSDeepTauPFTauSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..modules.hltL1SingleNNTau150_cfi import *
from ..modules.hltPreLooseDeepTauPFTauHPS150L2NNeta2p1_cfi import *
from ..modules.hltParticleFlowRecHitECALUnseeded_cfi import *
from ..modules.hltParticleFlowClusterECALUncorrectedUnseeded_cfi import *
from ..modules.hltParticleFlowClusterECALUnseeded_cfi import *
from ..modules.hltAK4PFJetsForTaus_cfi import *
from ..modules.hltHpsSelectedPFTauLooseTauWPDeepTau_cfi import *
from ..modules.hltHpsPFTau150LooseTauWPDeepTau_cfi import *

HLT_LooseDeepTauPFTauHPS150_L2NN_eta2p1 = cms.Path(
    HLTBeginSequence 
    + hltL1SingleNNTau150
    + hltPreLooseDeepTauPFTauHPS150L2NNeta2p1                               
    + HLTRawToDigiSequence 
    + HLTTICLLocalRecoSequence 
    + HLTLocalrecoSequence 
    + HLTTrackingSequence 
    + HLTMuonsSequence 
    + HLTParticleFlowSequence 
    + hltParticleFlowRecHitECALUnseeded
    + hltParticleFlowClusterECALUncorrectedUnseeded
    + hltParticleFlowClusterECALUnseeded
    + HLTAK4PFJetsReconstruction 
    + hltAK4PFJetsForTaus 
    + HLTPFTauHPS 
    + HLTHPSDeepTauPFTauSequence 
    + hltHpsSelectedPFTauLooseTauWPDeepTau 
    + hltHpsPFTau150LooseTauWPDeepTau 
    + HLTEndSequence 
)


