import FWCore.ParameterSet.Config as cms

# Phase-2 port of the Run-3 VBF + MET path HLT_VBF_DiPFJet80_45_Mjj650_PFMETNoMu85_v*
# For now, thresholds aligned to the v0p0_2026
# L1 seed pDoubleJet_METSum.

from ..modules.hltL1SeedsForVBFDiPFPuppiJetFilter_cfi import *
from ..modules.hltPFPuppiMETTypeOneCorrector_cfi import *
from ..modules.hltPFPuppiMETTypeOne_cfi import *
from ..modules.hltPFPuppiMETTypeOne176_cfi import *
from ..modules.hltDiPFPuppiJet83MJJ600_cfi import *
from ..modules.hltSinglePFPuppiJet118MaxEta5_cfi import *
from ..modules.hltPFPuppiJetCategoriesVBFMET600_cfi import *
from ..sequences.HLTTICLLocalRecoSequence_cfi import *
from ..sequences.HLTAK4PFPuppiJetsReconstruction_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.HLTMuonsSequence_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *
from ..sequences.HLTPFPuppiMETReconstruction_cfi import *
from ..sequences.HLTTrackingSequence_cfi import *
from ..sequences.HLTLocalrecoSequence_cfi import *
from ..sequences.HLTRawToDigiSequence_cfi import *

HLT_VBF_DiPFPuppiJet118_83_Mjj600_PFPuppiMETTypeOne176 = cms.Path(
    HLTBeginSequence
    + hltL1SeedsForVBFDiPFPuppiJetFilter
    + HLTRawToDigiSequence
    + HLTLocalrecoSequence
    + HLTTICLLocalRecoSequence
    + HLTTrackingSequence
    + HLTMuonsSequence
    + HLTParticleFlowSequence
    + HLTAK4PFPuppiJetsReconstruction
    + HLTPFPuppiMETReconstruction
    + hltPFPuppiMETTypeOneCorrector
    + hltPFPuppiMETTypeOne
    + hltPFPuppiMETTypeOne176
    + hltDiPFPuppiJet83MJJ600
    + hltSinglePFPuppiJet118MaxEta5
    + hltPFPuppiJetCategoriesVBFMET600
    + HLTEndSequence
)
