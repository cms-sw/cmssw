import FWCore.ParameterSet.Config as cms

# Phase-2 port of the Run-3 VBF + single-tau path HLT_VBF_DiPFJet45_Mjj650_PNetTauhPFJet45_L2NN_eta2p3_v*
# For now, thresholds aligned to the v0p0_2026
# L1 seed pDoubleJet_IsoTau.

from ..modules.hltL1SeedsForVBFDiPFPuppiJetFilter_cfi import *
from ..sequences.HLTRawToDigiSequence_cfi import *
from ..sequences.HLTLocalrecoSequence_cfi import *
from ..sequences.HLTTICLLocalRecoSequence_cfi import *
from ..sequences.HLTTrackingSequence_cfi import *
from ..sequences.HLTMuonsSequence_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *
from ..sequences.HLTAK4PFPuppiJetsReconstruction_cfi import *
from ..modules.hltDiPFPuppiJet93MJJ600_cfi import *
from ..modules.hltPFPuppiJetCategoriesVBF93Mjj600_cfi import *
from ..modules.hltSinglePFPuppiJet61MaxEta2p1ForParTTau_cfi import *
from ..sequences.HLTJetFlavourTagParticleTransformerSequencePF_cfi import *
from ..modules.hltSinglePFPuppiJet61ParTTauhTagMediumWP_cfi import *
from ..modules.hltDiPFPuppiJet93ParTTauh61OverlapFilter_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

HLT_VBF_DiPFPuppiJet93_Mjj600_MediumPFPuppiParTTauh61_eta2p1 = cms.Path(
    HLTBeginSequence
    + hltL1SeedsForVBFDiPFPuppiJetFilter
    + HLTRawToDigiSequence
    + HLTLocalrecoSequence
    + HLTTICLLocalRecoSequence
    + HLTTrackingSequence
    + HLTMuonsSequence
    + HLTParticleFlowSequence
    + HLTAK4PFPuppiJetsReconstruction
    + hltDiPFPuppiJet93MJJ600
    + hltPFPuppiJetCategoriesVBF93Mjj600
    + hltSinglePFPuppiJet61MaxEta2p1ForParTTau
    + HLTJetFlavourTagParticleTransformerSequencePF
    + hltSinglePFPuppiJet61ParTTauhTagMediumWP
    + hltDiPFPuppiJet93ParTTauh61OverlapFilter
    + HLTEndSequence
)
