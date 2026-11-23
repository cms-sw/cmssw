import FWCore.ParameterSet.Config as cms

# Phase-2 port of the Run-3 VBF + double-tau path HLT_VBF_DiPFJet115_40_Mjj850_DoublePNetTauhPFJet20_eta2p3_v*
# For now, thresholds aligned to the v0p0_2026
# L1 seed pDoubleJet_Mass_Min1000 (no VBF + double-tau seed).

from ..modules.hltL1SeedsForVBFDiPFPuppiJetFilter_cfi import *
from ..sequences.HLTRawToDigiSequence_cfi import *
from ..sequences.HLTLocalrecoSequence_cfi import *
from ..sequences.HLTTICLLocalRecoSequence_cfi import *
from ..sequences.HLTTrackingSequence_cfi import *
from ..sequences.HLTMuonsSequence_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *
from ..sequences.HLTAK4PFPuppiJetsReconstruction_cfi import *
from ..modules.hltDiPFPuppiJet70MJJ1000_cfi import *
from ..modules.hltPFPuppiJetCategoriesVBF1000_cfi import *
from ..modules.hltDoublePFPuppiJets20MaxEta2p1ForParTTau_cfi import *
from ..sequences.HLTJetFlavourTagParticleTransformerSequencePF_cfi import *
from ..modules.hltDoublePFPuppiJets20ParTTauhTagMediumWP_cfi import *
from ..modules.hltDiPFPuppiJet70DoubleParTTauh20OverlapFilter_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

HLT_VBF_DiPFPuppiJet145_70_Mjj1000_DoubleMediumPFPuppiParTTauh20_eta2p1 = cms.Path(
    HLTBeginSequence
    + hltL1SeedsForVBFDiPFPuppiJetFilter
    + HLTRawToDigiSequence
    + HLTLocalrecoSequence
    + HLTTICLLocalRecoSequence
    + HLTTrackingSequence
    + HLTMuonsSequence
    + HLTParticleFlowSequence
    + HLTAK4PFPuppiJetsReconstruction
    + hltDiPFPuppiJet70MJJ1000
    + hltPFPuppiJetCategoriesVBF1000
    + hltDoublePFPuppiJets20MaxEta2p1ForParTTau
    + HLTJetFlavourTagParticleTransformerSequencePF
    + hltDoublePFPuppiJets20ParTTauhTagMediumWP
    + hltDiPFPuppiJet70DoubleParTTauh20OverlapFilter
    + HLTEndSequence
)
