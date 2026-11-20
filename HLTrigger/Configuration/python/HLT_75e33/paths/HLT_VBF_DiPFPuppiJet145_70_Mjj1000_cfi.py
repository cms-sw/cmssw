import FWCore.ParameterSet.Config as cms

# Phase-2 port of the Run-3 inclusive VBF path HLT_VBF_DiPFJet125_45_Mjj1050_v*
# For now, thresholds aligned to the v0p0_2026
# L1 seed pDoubleJet_Mass_Min1000.

from ..modules.hltL1SeedsForVBFDiPFPuppiJetFilter_cfi import *
from ..modules.hltDoublePFPuppiJets70MaxEta5p2_cfi import *
from ..modules.hltSinglePFPuppiJet145MaxEta5p2_cfi import *
from ..modules.hltDiPFPuppiJet70MJJ1000_cfi import *
from ..modules.hltPFPuppiJetCategoriesVBF1000_cfi import *
from ..sequences.HLTTICLLocalRecoSequence_cfi import *
from ..sequences.HLTAK4PFPuppiJetsReconstruction_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.HLTMuonsSequence_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *
from ..sequences.HLTTrackingSequence_cfi import *
from ..sequences.HLTLocalrecoSequence_cfi import *
from ..sequences.HLTRawToDigiSequence_cfi import *

HLT_VBF_DiPFPuppiJet145_70_Mjj1000 = cms.Path(
    HLTBeginSequence
    + hltL1SeedsForVBFDiPFPuppiJetFilter
    + HLTRawToDigiSequence
    + HLTLocalrecoSequence
    + HLTTICLLocalRecoSequence
    + HLTTrackingSequence
    + HLTMuonsSequence
    + HLTParticleFlowSequence
    + HLTAK4PFPuppiJetsReconstruction
    + hltDoublePFPuppiJets70MaxEta5p2
    + hltSinglePFPuppiJet145MaxEta5p2
    + hltDiPFPuppiJet70MJJ1000
    + hltPFPuppiJetCategoriesVBF1000
    + HLTEndSequence
)
