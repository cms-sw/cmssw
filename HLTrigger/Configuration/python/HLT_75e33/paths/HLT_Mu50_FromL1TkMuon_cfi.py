import FWCore.ParameterSet.Config as cms

from ..modules.hltSingleTkMuon22L1TkMuonFilter_cfi import *
from ..modules.hltL3fL1TkSingleMu22L3Filtered50Q_cfi import *
from ..modules.hltPhase2L3MuonCandidates_cfi import *
from ..modules.hltPhase2PixelFitterByHelixProjections_cfi import *
from ..modules.hltPhase2PixelTrackFilterByKinematics_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.HLTMuonsSequence_cfi import *
from ..sequences.HLTTrackingV61Sequence_cfi import *
from ..sequences.HLTItLocalRecoSequence_cfi import *
from ..sequences.HLTMuonlocalrecoSequence_cfi import *
from ..sequences.HLTOtLocalRecoSequence_cfi import *

HLT_Mu50_FromL1TkMuon = cms.Path(
    HLTBeginSequence
    + hltSingleTkMuon22L1TkMuonFilter
    + HLTMuonlocalrecoSequence
    + HLTItLocalRecoSequence
    + HLTOtLocalRecoSequence
    + hltPhase2PixelFitterByHelixProjections
    + hltPhase2PixelTrackFilterByKinematics
    + HLTMuonsSequence
    + hltPhase2L3MuonCandidates
    + hltL3fL1TkSingleMu22L3Filtered50Q
    + HLTEndSequence
)
