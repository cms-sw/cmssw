import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.Skimming.dimuonsHLTFilter_cfi import *
from ElectroWeakAnalysis.Skimming.patCandidatesForZMuMuSubskim_cff import *
from ElectroWeakAnalysis.ZMuMu.zMuMuMuonUserData import *
from ElectroWeakAnalysis.ZMuMu.dimuonsUserData_cfi import *
from ElectroWeakAnalysis.ZMuMu.dimuonsOneTrackUserData_cfi import *
from ElectroWeakAnalysis.ZMuMu.dimuonsGlobal_cfi import *
from ElectroWeakAnalysis.ZMuMu.dimuonsOneStandAloneMuonUserData_cfi import *
from ElectroWeakAnalysis.ZMuMu.dimuonsFilter_cfi import *
from ElectroWeakAnalysis.ZMuMu.dimuonsOneTrackFilterUserData_cfi import *

dimuonsPath = cms.Path(
    dimuonsHLTFilter *
    goodMuonRecoForDimuon *
    userDataMuons*
    dimuons *
    userDataDimuons*
    dimuonsGlobal *
    dimuonsOneStandAloneMuon *
    dimuonsFilter    
)

dimuonsOneTrackPath = cms.Path(dimuonsHLTFilter+
                               goodMuonRecoForDimuon*
                               userDataMuons*
                               userDataTracks*
                               dimuonsOneTrack*
                               userDataDimuonsOneTrack*
                               dimuonsOneTrackFilter
)


