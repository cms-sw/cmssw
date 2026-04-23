import FWCore.ParameterSet.Config as cms

# Phase-2 port of the Run-3 VBF + muon path HLT_VBF_DiPFJet95_45_Mjj750_Mu3_TrkIsoVVL_v*
# For now, thresholds aligned to the v0p0_2026
# L1 seed pDoubleJet_TkMu.

from ..modules.hltL1SeedsForVBFDiPFPuppiJetFilter_cfi import *
from ..sequences.HLTRawToDigiSequence_cfi import *
from ..sequences.HLTLocalrecoSequence_cfi import *
from ..sequences.HLTTICLLocalRecoSequence_cfi import *
from ..sequences.HLTTrackingSequence_cfi import *
from ..sequences.HLTMuonsSequence_cfi import *
from ..modules.hltL3fL1TkMuVBFFiltered5_cfi import *
from ..sequences.HLTPhase2L3MuonIsoTracksSequence_cfi import *
from ..modules.hltPhase2L3MuonsTrkIsoRegionalNewdR0p3dRVeto0p005dz0p25dr0p20ChisqInfPtMin0p0Cut0p4_cfi import *
from ..modules.hltL3fL1TkMuVBF5RelTrkIsoVVLFiltered0p4_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *
from ..sequences.HLTAK4PFPuppiJetsReconstruction_cfi import *
from ..modules.hltDiPFPuppiJet83MJJ700_cfi import *
from ..modules.hltSinglePFPuppiJet133MaxEta5_cfi import *
from ..modules.hltPFPuppiJetCategoriesVBFMu700_cfi import *
from ..modules.hltDiPFPuppiJet83Mu5OverlapFilter_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

HLT_VBF_DiPFPuppiJet133_83_Mjj700_Mu5_TrkIsoVVL = cms.Path(
    HLTBeginSequence
    + hltL1SeedsForVBFDiPFPuppiJetFilter
    + HLTRawToDigiSequence
    + HLTLocalrecoSequence
    + HLTTICLLocalRecoSequence
    + HLTTrackingSequence
    + HLTMuonsSequence
    + hltL3fL1TkMuVBFFiltered5
    + HLTPhase2L3MuonIsoTracksSequence
    + hltPhase2L3MuonsTrkIsoRegionalNewdR0p3dRVeto0p005dz0p25dr0p20ChisqInfPtMin0p0Cut0p4
    + hltL3fL1TkMuVBF5RelTrkIsoVVLFiltered0p4
    + HLTParticleFlowSequence
    + HLTAK4PFPuppiJetsReconstruction
    + hltDiPFPuppiJet83MJJ700
    + hltSinglePFPuppiJet133MaxEta5
    + hltPFPuppiJetCategoriesVBFMu700
    + hltDiPFPuppiJet83Mu5OverlapFilter
    + HLTEndSequence
)
