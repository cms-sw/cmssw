import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.Skimming.dimuonsHLTFilter_cfi import *
from ElectroWeakAnalysis.Skimming.patCandidatesForZMuMuSubskim_cff import *
from ElectroWeakAnalysis.Skimming.dimuons_cfi import *
from ElectroWeakAnalysis.Skimming.dimuonsOneTrack_cfi import *
from ElectroWeakAnalysis.Skimming.dimuonsGlobal_cfi import *
from ElectroWeakAnalysis.Skimming.dimuonsOneStandAloneMuon_cfi import *
from ElectroWeakAnalysis.Skimming.dimuonsOneTrackerMuon_cfi import *
from ElectroWeakAnalysis.Skimming.dimuonsFilter_cfi import *
from ElectroWeakAnalysis.Skimming.dimuonsOneTrackFilter_cfi import *

dimuonsPath = cms.Path(
    dimuonsHLTFilter *
    goodMuonRecoForDimuon *
    dimuons *
    dimuonsGlobal *
    dimuonsOneStandAloneMuon *
    dimuonsOneTrackerMuon *
    dimuonsFilter    
)

dimuonsOneTrackPath = cms.Path(dimuonsHLTFilter+
                               goodMuonRecoForDimuon+
                               dimuonsOneTrack+
                               dimuonsOneTrackFilter
)


