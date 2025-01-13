import FWCore.ParameterSet.Config as cms

from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.HLTRawToDigiSequence_cfi import *
from ..sequences.HLTItLocalRecoSequence_cfi import *
from ..sequences.HLTOtLocalRecoSequence_cfi import *
from ..sequences.HLTMuonsSequence_cfi import *
from ..modules.hltPhase2PixelFitterByHelixProjections_cfi import *
from ..modules.hltPhase2PixelTrackFilterByKinematics_cfi import *
from ..modules.hltL3fL1TkTripleMu533L31055DZFiltered0p2_cfi import *
from ..modules.hltL3fL1TkTripleMu533L3Filtered1055_cfi import *
from ..modules.hltL3fL1TkTripleMu533PreFiltered555_cfi import *
from ..modules.hltTripleMuon3DR0_cfi import *
from ..modules.hltTripleMuon3DZ1p0_cfi import *

HLT_TriMu_10_5_5_DZ_FromL1TkMuon = cms.Path(
    HLTBeginSequence
    + hltTripleMuon3DZ1p0
    + hltTripleMuon3DR0
    + HLTRawToDigiSequence
    + HLTItLocalRecoSequence
    + HLTOtLocalRecoSequence
    + hltPhase2PixelFitterByHelixProjections
    + hltPhase2PixelTrackFilterByKinematics
    + HLTMuonsSequence
    + hltL3fL1TkTripleMu533PreFiltered555
    + hltL3fL1TkTripleMu533L3Filtered1055
    + hltL3fL1TkTripleMu533L31055DZFiltered0p2
    + HLTEndSequence
)

