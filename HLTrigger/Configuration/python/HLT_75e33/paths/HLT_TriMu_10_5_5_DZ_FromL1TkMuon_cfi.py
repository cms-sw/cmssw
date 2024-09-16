import FWCore.ParameterSet.Config as cms

from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.HLTRawToDigiSequence_cfi import *
from ..sequences.HLTItLocalRecoSequence_cfi import *
from ..sequences.HLTOtLocalRecoSequence_cfi import *
from ..sequences.HLTHgcalLocalRecoSequence_cfi import *
from ..sequences.HLTDoLocalHcalSequence_cfi import *
from ..sequences.HLTDoFullUnpackingEgammaEcalSequence_cfi import *
from ..sequences.HLTFastJetForEgammaSequence_cfi import *
from ..sequences.HLTIter0Phase2L3FromL1TkSequence_cfi import *
from ..sequences.HLTIter2Phase2L3FromL1TkSequence_cfi import *
from ..sequences.HLTPfClusteringHBHEHFSequence_cfi import *
from ..sequences.HLTPhase2L3FromL1TkSequence_cfi import *
from ..sequences.HLTPhase2L3OISequence_cfi import *
from ..sequences.HLTPhase2L3MuonsSequence_cfi import *
from ..sequences.HLTL2MuonsFromL1TkSequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaUnseededSequence_cfi import *
from ..modules.hltPhase2PixelFitterByHelixProjections_cfi import *
from ..modules.hltPhase2PixelTrackFilterByKinematics_cfi import *
from ..modules.hltL3fL1TkTripleMu533L31055DZFiltered0p2_cfi import *
from ..modules.hltL3fL1TkTripleMu533L3Filtered1055_cfi import *
from ..modules.hltL3fL1TkTripleMu533PreFiltered555_cfi import *
from ..modules.hltTripleMuon3DR0_cfi import *
from ..modules.hltTripleMuon3DZ1p0_cfi import *

HLT_TriMu_10_5_5_DZ_FromL1TkMuon = cms.Path(HLTBeginSequence
    +hltTripleMuon3DZ1p0
    +hltTripleMuon3DR0
    +HLTRawToDigiSequence
    +HLTItLocalRecoSequence
    +HLTOtLocalRecoSequence
    +HLTL2MuonsFromL1TkSequence
    +HLTPhase2L3MuonsSequence
    +hltL3fL1TkTripleMu533PreFiltered555
    +hltL3fL1TkTripleMu533L3Filtered1055
    +HLTPhase2L3FromL1TkSequence
    +hltPhase2PixelFitterByHelixProjections
    +hltPhase2PixelTrackFilterByKinematics
    +HLTIter0Phase2L3FromL1TkSequence
    +HLTIter2Phase2L3FromL1TkSequence
    +HLTPhase2L3OISequence
    +hltL3fL1TkTripleMu533L31055DZFiltered0p2
    +HLTEndSequence)
#
