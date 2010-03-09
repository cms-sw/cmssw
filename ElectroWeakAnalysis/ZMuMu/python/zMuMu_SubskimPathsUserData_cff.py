import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.Skimming.dimuonsHLTFilter_cfi import *
from ElectroWeakAnalysis.Skimming.patCandidatesForZMuMuSubskim_cff import *
from ElectroWeakAnalysis.ZMuMu.zMuMuMuonUserData import *
from ElectroWeakAnalysis.ZMuMu.dimuonsUserData_cfi import *
from ElectroWeakAnalysis.ZMuMu.dimuonsOneTrackUserData_cfi import *
from ElectroWeakAnalysis.Skimming.dimuonsGlobal_cfi import *
from ElectroWeakAnalysis.Skimming.dimuonsOneStandAloneMuon_cfi import *
from ElectroWeakAnalysis.Skimming.dimuonsFilter_cfi import *
from ElectroWeakAnalysis.Skimming.dimuonsOneTrackFilter_cfi import *

dimuonsPath = cms.Path(
    dimuonsHLTFilter *
    goodMuonRecoForDimuon *
    userDataMuons*
    dimuons *
    dimuonsGlobal *
    dimuonsOneStandAloneMuon *
    dimuonsFilter    
)

dimuonsOneTrackPath = cms.Path(dimuonsHLTFilter+
                               goodMuonRecoForDimuon*
                               userDataMuons*
                               userDataTracks*
                               dimuonsOneTrack*
                               dimuonsOneTrackFilter
)


