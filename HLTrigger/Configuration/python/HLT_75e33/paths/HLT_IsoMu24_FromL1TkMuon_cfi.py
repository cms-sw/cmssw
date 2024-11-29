import FWCore.ParameterSet.Config as cms

from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.HLTRawToDigiSequence_cfi import *
from ..sequences.HLTItLocalRecoSequence_cfi import *
from ..sequences.HLTOtLocalRecoSequence_cfi import *
from ..sequences.HLTHgcalLocalRecoSequence_cfi import *
from ..sequences.HLTDoLocalHcalSequence_cfi import *
from ..sequences.HLTDoFullUnpackingEgammaEcalSequence_cfi import *
from ..sequences.HLTMuonsSequence_cfi import *
from ..sequences.HLTFastJetForEgammaSequence_cfi import *
from ..sequences.HLTPfClusteringHBHEHFSequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaUnseededSequence_cfi import *
from ..sequences.HLTPhase2L3MuonGeneralTracksSequence_cfi import *
from ..modules.hltSingleTkMuon22L1TkMuonFilter_cfi import *
from ..modules.hltPhase2PixelFitterByHelixProjections_cfi import *
from ..modules.hltPhase2PixelTrackFilterByKinematics_cfi import *
from ..modules.hltL3crIsoL1TkSingleMu22L3f24QL3pfecalIsoFiltered0p41_cfi import *
from ..modules.hltL3crIsoL1TkSingleMu22L3f24QL3pfhcalIsoFiltered0p40_cfi import *
from ..modules.hltL3crIsoL1TkSingleMu22L3f24QL3pfhgcalIsoFiltered4p70_cfi import *
from ..modules.hltL3crIsoL1TkSingleMu22L3f24QL3trkIsoRegionalNewFiltered0p07EcalHcalHgcalTrk_cfi import *
from ..modules.hltL3fL1TkSingleMu22L3Filtered24Q_cfi import *
from ..modules.hltPhase2L3MuonsEcalIsodR0p3dRVeto0p000_cfi import *
from ..modules.hltPhase2L3MuonsHcalIsodR0p3dRVeto0p000_cfi import *
from ..modules.hltPhase2L3MuonsHgcalLCIsodR0p2dRVetoEM0p00dRVetoHad0p02minEEM0p00minEHad0p00_cfi import *
from ..modules.hltPhase2L3MuonsTrkIsoRegionalNewdR0p3dRVeto0p005dz0p25dr0p20ChisqInfPtMin0p0Cut0p07_cfi import *

HLT_IsoMu24_FromL1TkMuon = cms.Path(
    HLTBeginSequence
    + hltSingleTkMuon22L1TkMuonFilter
    + HLTRawToDigiSequence
    + HLTItLocalRecoSequence
    + HLTOtLocalRecoSequence
    + hltPhase2PixelFitterByHelixProjections
    + hltPhase2PixelTrackFilterByKinematics
    + HLTMuonsSequence
    + hltL3fL1TkSingleMu22L3Filtered24Q
    + HLTHgcalLocalRecoSequence
    + HLTDoLocalHcalSequence
    + HLTDoFullUnpackingEgammaEcalSequence
    + HLTFastJetForEgammaSequence
    + HLTPfClusteringHBHEHFSequence
    + HLTPFClusteringForEgammaUnseededSequence
    + hltPhase2L3MuonsEcalIsodR0p3dRVeto0p000
    + hltPhase2L3MuonsHcalIsodR0p3dRVeto0p000
    + hltPhase2L3MuonsHgcalLCIsodR0p2dRVetoEM0p00dRVetoHad0p02minEEM0p00minEHad0p00
    + hltL3crIsoL1TkSingleMu22L3f24QL3pfecalIsoFiltered0p41
    + hltL3crIsoL1TkSingleMu22L3f24QL3pfhcalIsoFiltered0p40
    + hltL3crIsoL1TkSingleMu22L3f24QL3pfhgcalIsoFiltered4p70
    + HLTPhase2L3MuonGeneralTracksSequence
    + hltPhase2L3MuonsTrkIsoRegionalNewdR0p3dRVeto0p005dz0p25dr0p20ChisqInfPtMin0p0Cut0p07
    + hltL3crIsoL1TkSingleMu22L3f24QL3trkIsoRegionalNewFiltered0p07EcalHcalHgcalTrk
    + HLTEndSequence
)
