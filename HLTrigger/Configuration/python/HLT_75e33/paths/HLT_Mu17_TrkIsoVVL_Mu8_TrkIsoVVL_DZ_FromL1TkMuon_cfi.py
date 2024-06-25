import FWCore.ParameterSet.Config as cms

from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.RawToDigiSequence_cfi import *
from ..sequences.itLocalRecoSequence_cfi import *
from ..sequences.otLocalRecoSequence_cfi import *
from ..sequences.hgcalLocalRecoSequence_cfi import *
from ..sequences.HLTDoLocalHcalSequence_cfi import *
from ..sequences.HLTDoFullUnpackingEgammaEcalSequence_cfi import *
from ..sequences.HLTFastJetForEgammaSequence_cfi import *
from ..sequences.HLTIter0Phase2L3FromL1TkSequence_cfi import *
from ..sequences.HLTIter2Phase2L3FromL1TkSequence_cfi import *
from ..sequences.pfClusteringHBHEHFSequence_cfi import *
from ..sequences.HLTPhase2L3FromL1TkSequence_cfi import *
from ..sequences.HLTPhase2L3OISequence_cfi import *
from ..sequences.HLTPhase2L3MuonsSequence_cfi import *
from ..sequences.HLTL2MuonsFromL1TkSequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaUnseededSequence_cfi import *
from ..sequences.HLTPhase2L3MuonGeneralTracksSequence_cfi import *
from ..modules.hltDoubleTkMuon157L1TkMuonFilter_cfi import *
from ..modules.hltPhase2PixelFitterByHelixProjections_cfi import *
from ..modules.hltPhase2PixelTrackFilterByKinematics_cfi import *
from ..modules.hltDiMuon178RelTrkIsoFiltered0p4_cfi import *
from ..modules.hltDiMuon178RelTrkIsoFiltered0p4DzFiltered0p2_cfi import *
from ..modules.hltDoubleMuon7DZ1p0_cfi import *
from ..modules.hltL1TkDoubleMuFiltered7_cfi import *
from ..modules.hltL1TkSingleMuFiltered15_cfi import *
from ..modules.hltL3fL1DoubleMu155fFiltered17_cfi import *
from ..modules.hltL3fL1DoubleMu155fPreFiltered8_cfi import *
from ..modules.hltPhase2L3MuonsTrkIsoRegionalNewdR0p3dRVeto0p005dz0p25dr0p20ChisqInfPtMin0p0Cut0p4_cfi import *

HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_FromL1TkMuon = cms.Path(HLTBeginSequence
    +hltDoubleTkMuon157L1TkMuonFilter
    +hltDoubleMuon7DZ1p0
    +RawToDigiSequence
    +itLocalRecoSequence
    +otLocalRecoSequence
    +HLTL2MuonsFromL1TkSequence
    +HLTPhase2L3OISequence
    +hltPhase2PixelFitterByHelixProjections
    +hltPhase2PixelTrackFilterByKinematics
    +HLTPhase2L3FromL1TkSequence
    +HLTIter0Phase2L3FromL1TkSequence
    +HLTIter2Phase2L3FromL1TkSequence
    +HLTPhase2L3MuonsSequence
    +hltL3fL1DoubleMu155fPreFiltered8
    +hltL3fL1DoubleMu155fFiltered17
    +HLTPhase2L3MuonGeneralTracksSequence
    +hltPhase2L3MuonsTrkIsoRegionalNewdR0p3dRVeto0p005dz0p25dr0p20ChisqInfPtMin0p0Cut0p4
    +hltDiMuon178RelTrkIsoFiltered0p4
    +hltDiMuon178RelTrkIsoFiltered0p4DzFiltered0p2
    +HLTEndSequence)
