import FWCore.ParameterSet.Config as cms

# Phase-2 port of the Run-3 VBF + electron path HLT_VBF_DiPFJet50_Mjj600_Ele22_eta2p1_WPTight_Gsf_v*
# For now, thresholds aligned to the v0p0_2026
# L1 seed pDoubleJet_TkEleIso.

from ..modules.hltL1SeedsForVBFDiPFPuppiJetFilter_cfi import *
from ..sequences.HLTEle24erWPTightVBFL1SeededSequence_cfi import *
from ..sequences.HLTRawToDigiSequence_cfi import *
from ..sequences.HLTLocalrecoSequence_cfi import *
from ..sequences.HLTTICLLocalRecoSequence_cfi import *
from ..sequences.HLTTrackingSequence_cfi import *
from ..sequences.HLTMuonsSequence_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *
from ..sequences.HLTAK4PFPuppiJetsReconstruction_cfi import *
from ..modules.hltDiPFPuppiJet93MJJ550_cfi import *
from ..modules.hltPFPuppiJetCategoriesVBF93Mjj550_cfi import *
from ..modules.hltDiPFPuppiJet93Ele24OverlapFilter_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

HLT_VBF_DiPFPuppiJet93_Mjj550_Ele24_eta2p1_WPTight_L1Seeded = cms.Path(
    HLTBeginSequence
    + hltL1SeedsForVBFDiPFPuppiJetFilter
    + HLTEle24erWPTightVBFL1SeededSequence
    + HLTRawToDigiSequence
    + HLTLocalrecoSequence
    + HLTTICLLocalRecoSequence
    + HLTTrackingSequence
    + HLTMuonsSequence
    + HLTParticleFlowSequence
    + HLTAK4PFPuppiJetsReconstruction
    + hltDiPFPuppiJet93MJJ550
    + hltPFPuppiJetCategoriesVBF93Mjj550
    + hltDiPFPuppiJet93Ele24OverlapFilter
    + HLTEndSequence
)
