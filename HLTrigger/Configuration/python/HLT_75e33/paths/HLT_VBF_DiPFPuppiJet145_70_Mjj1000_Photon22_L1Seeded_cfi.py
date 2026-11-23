import FWCore.ParameterSet.Config as cms

# Phase-2 port of the Run-3 VBF + photon path HLT_VBF_DiPFJet50_Mjj650_Photon22_v*
# For now, thresholds aligned to the v0p0_2026
# L1 seed pDoubleJet_Mass_Min1000 (no dedicated VBF + e/gamma seed).

from ..modules.hltL1SeedsForVBFDiPFPuppiJetFilter_cfi import *
from ..sequences.HLTPhoton22VBFL1SeededSequence_cfi import *
from ..sequences.HLTRawToDigiSequence_cfi import *
from ..sequences.HLTLocalrecoSequence_cfi import *
from ..sequences.HLTTICLLocalRecoSequence_cfi import *
from ..sequences.HLTTrackingSequence_cfi import *
from ..sequences.HLTMuonsSequence_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *
from ..sequences.HLTAK4PFPuppiJetsReconstruction_cfi import *
from ..modules.hltDiPFPuppiJet70MJJ1000_cfi import *
from ..modules.hltPFPuppiJetCategoriesVBF1000_cfi import *
from ..modules.hltDiPFPuppiJet70Photon22OverlapFilter_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

HLT_VBF_DiPFPuppiJet145_70_Mjj1000_Photon22_L1Seeded = cms.Path(
    HLTBeginSequence
    + hltL1SeedsForVBFDiPFPuppiJetFilter
    + HLTPhoton22VBFL1SeededSequence
    + HLTRawToDigiSequence
    + HLTLocalrecoSequence
    + HLTTICLLocalRecoSequence
    + HLTTrackingSequence
    + HLTMuonsSequence
    + HLTParticleFlowSequence
    + HLTAK4PFPuppiJetsReconstruction
    + hltDiPFPuppiJet70MJJ1000
    + hltPFPuppiJetCategoriesVBF1000
    + hltDiPFPuppiJet70Photon22OverlapFilter
    + HLTEndSequence
)
