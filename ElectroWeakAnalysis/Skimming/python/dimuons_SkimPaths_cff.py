import FWCore.ParameterSet.Config as cms

# migration to PAT v2 and reorganization of dimuon sequences

from ElectroWeakAnalysis.Skimming.dimuonsHLTFilter_cfi import *
from ElectroWeakAnalysis.Skimming.patCandidatesForDimuonsSequences_cff import *
from ElectroWeakAnalysis.Skimming.dimuons_cfi import *
from ElectroWeakAnalysis.Skimming.dimuonsOneTrack_cfi import *
from ElectroWeakAnalysis.Skimming.dimuonsGlobal_cfi import *
from ElectroWeakAnalysis.Skimming.dimuonsOneStandAloneMuon_cfi import *
from ElectroWeakAnalysis.Skimming.mcTruthForDimuons_cff import *
from ElectroWeakAnalysis.Skimming.dimuonsFilter_cfi import *
from ElectroWeakAnalysis.Skimming.dimuonsOneTrackFilter_cfi import *

dimuonsPath = cms.Path(
    dimuonsHLTFilter *
    goodMuonRecoForDimuon *
    dimuons *
    dimuonsGlobal *
    dimuonsOneStandAloneMuon *
    dimuonsFilter    
)

dimuonsOneTrackPath = cms.Path(dimuonsHLTFilter+
                               goodMuonRecoForDimuon+
                               dimuonsOneTrack+
                               dimuonsOneTrackFilter
)

dimuonsMCTruth = cms.Path(dimuonsHLTFilter+
                          mcTruthForDimuons
)

