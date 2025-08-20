import FWCore.ParameterSet.Config as cms

from ..modules.hltDoubleMuon7DZ1p0_cfi import *
from ..modules.hltL3fL1DoubleMu155fFiltered37_cfi import *
from ..modules.hltL3fL1DoubleMu155fPreFiltered27_cfi import *
from ..modules.hltPhase2L3MuonCandidates_cfi import *
from ..modules.hltPhase2PixelFitterByHelixProjections_cfi import *
from ..modules.hltPhase2PixelTrackFilterByKinematics_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.HLTMuonsSequence_cfi import *
from ..sequences.HLTItLocalRecoSequence_cfi import *
from ..sequences.HLTMuonlocalrecoSequence_cfi import *
from ..sequences.HLTOtLocalRecoSequence_cfi import *

HLT_Mu37_Mu27_FromL1TkMuon = cms.Path(
    HLTBeginSequence
    + hltDoubleMuon7DZ1p0
    + HLTMuonlocalrecoSequence
    + HLTItLocalRecoSequence
    + HLTOtLocalRecoSequence
    + hltPhase2PixelFitterByHelixProjections
    + hltPhase2PixelTrackFilterByKinematics
    + HLTMuonsSequence
    + hltPhase2L3MuonCandidates
    + hltL3fL1DoubleMu155fPreFiltered27
    + hltL3fL1DoubleMu155fFiltered37
    + HLTEndSequence
)
