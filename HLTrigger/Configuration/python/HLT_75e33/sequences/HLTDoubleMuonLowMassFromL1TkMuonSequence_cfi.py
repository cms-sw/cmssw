import FWCore.ParameterSet.Config as cms

from ..modules.hltDoubleTkMuonLowMassL1TkMuonFilter_cfi import *
from ..modules.hltL3fL1DoubleTkMuonLowMassPreFiltered3_cfi import *
from ..modules.hltPhase2L3MuonCandidates_cfi import *
from ..modules.hltPhase2PixelFitterByHelixProjections_cfi import *
from ..modules.hltPhase2PixelTrackFilterByKinematics_cfi import *
from ..sequences.HLTItLocalRecoSequence_cfi import *
from ..sequences.HLTMuonlocalrecoSequence_cfi import *
from ..sequences.HLTMuonsSequence_cfi import *
from ..sequences.HLTOtLocalRecoSequence_cfi import *

# Common part of the HLT_DoubleMu4_3_{LowMass,Bs,Jpsi}_FromL1TkMuon paths: L1 gate on the
# low-mass double-muon seeds, Phase-2 L3 muon reconstruction (as in HLT_Mu37_Mu27_FromL1TkMuon),
# and two L3 muons matched to the L1 seed muons. The paths differ only by the dimuon mass window.
HLTDoubleMuonLowMassFromL1TkMuonSequence = cms.Sequence(
    hltDoubleTkMuonLowMassL1TkMuonFilter
    + HLTMuonlocalrecoSequence
    + HLTItLocalRecoSequence
    + HLTOtLocalRecoSequence
    + hltPhase2PixelFitterByHelixProjections
    + hltPhase2PixelTrackFilterByKinematics
    + HLTMuonsSequence
    + hltPhase2L3MuonCandidates
    + hltL3fL1DoubleTkMuonLowMassPreFiltered3
)
