import FWCore.ParameterSet.Config as cms

from ..modules.hltSingleTkMuonBarrelL1TkMuonFilter_cfi import *
from ..modules.hltL3fL1TkSingleMuBarrelL3Filtered9_cfi import *
from ..modules.hltL3fL1TkSingleMuBarrelIP6L3Filtered9_cfi import *
from ..modules.hltPhase2L3MuonCandidates_cfi import *
from ..modules.hltPhase2PixelFitterByHelixProjections_cfi import *
from ..modules.hltPhase2PixelTrackFilterByKinematics_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.HLTItLocalRecoSequence_cfi import *
from ..sequences.HLTMuonlocalrecoSequence_cfi import *
from ..sequences.HLTMuonsSequence_cfi import *
from ..sequences.HLTOtLocalRecoSequence_cfi import *

# Port of the Run-3 HLT_Mu9_Barrel_L1HP10_IP6 path (HLT_GRun): L1 gate on pSingleTkMu_Barrel
# (v0p0_2026 L1 menu only), Phase-2 L3 muon reconstruction (as in HLT_Mu50_FromL1TkMuon), one L3
# muon with pT > 9 GeV matched to the L1 seed muon and one with pT > 9 GeV and dxy significance > 6.
HLT_Mu9_Barrel_IP6_FromL1TkMuon = cms.Path(
    HLTBeginSequence
    + hltSingleTkMuonBarrelL1TkMuonFilter
    + HLTMuonlocalrecoSequence
    + HLTItLocalRecoSequence
    + HLTOtLocalRecoSequence
    + hltPhase2PixelFitterByHelixProjections
    + hltPhase2PixelTrackFilterByKinematics
    + HLTMuonsSequence
    + hltPhase2L3MuonCandidates
    + hltL3fL1TkSingleMuBarrelL3Filtered9
    + hltL3fL1TkSingleMuBarrelIP6L3Filtered9
    + HLTEndSequence
)
