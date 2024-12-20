import FWCore.ParameterSet.Config as cms

from ..modules.hltBTagPFPuppiDeepFlavour0p935DoubleEta2p4_cfi import *
from ..modules.hltDoublePFPuppiJets128Eta2p4MaxDeta1p6_cfi import *
from ..modules.hltDoublePFPuppiJets128MaxEta2p4_cfi import *
from ..modules.hltL1SeedsForDoublePuppiJetBtagFilter_cfi import *
from ..sequences.HLTHgcalLocalRecoSequence_cfi import *
from ..sequences.HLTAK4PFPuppiJetsReconstruction_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTBtagDeepFlavourSequencePFPuppiModEta2p4_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.HLTMuonsSequence_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *
from ..sequences.HLTTrackingV61Sequence_cfi import *
from ..sequences.HLTLocalrecoSequence_cfi import *
from ..sequences.HLTRawToDigiSequence_cfi import *

HLT_DoublePFPuppiJets128_DoublePFPuppiBTagDeepFlavour_2p4 = cms.Path(
    HLTBeginSequence
    + hltL1SeedsForDoublePuppiJetBtagFilter
    + HLTRawToDigiSequence
    + HLTHgcalLocalRecoSequence
    + HLTLocalrecoSequence
    + HLTTrackingV61Sequence
    + HLTMuonsSequence
    + HLTParticleFlowSequence
    + HLTAK4PFPuppiJetsReconstruction
    + hltDoublePFPuppiJets128MaxEta2p4
    + hltDoublePFPuppiJets128Eta2p4MaxDeta1p6
    + HLTBtagDeepFlavourSequencePFPuppiModEta2p4
    + hltBTagPFPuppiDeepFlavour0p935DoubleEta2p4
    + HLTEndSequence
)
