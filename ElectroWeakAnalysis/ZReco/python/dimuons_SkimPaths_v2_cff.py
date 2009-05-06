import FWCore.ParameterSet.Config as cms

# migration to PAT v2 and reorganization of dimuon sequences

from ElectroWeakAnalysis.ZReco.dimuonsHLTFilter_cfi import *
from ElectroWeakAnalysis.ZReco.patCandidatesForDimuonsSequences_cff import *
from ElectroWeakAnalysis.ZReco.dimuons_cfi import *
from ElectroWeakAnalysis.ZReco.dimuonsOneTrack_cfi import *
from ElectroWeakAnalysis.ZReco.dimuonsGlobal_cfi import *
from ElectroWeakAnalysis.ZReco.dimuonsOneStandAloneMuon_cfi import *
from ElectroWeakAnalysis.ZReco.mcTruthForDimuons_cff import *
from ElectroWeakAnalysis.ZReco.dimuonsFilter_cfi import *
from ElectroWeakAnalysis.ZReco.dimuonsOneTrackFilter_cfi import *

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

