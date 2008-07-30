import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZReco.dimuonsSequences_cff import *
from ElectroWeakAnalysis.ZReco.dimuonsHLTFilter_cfi import *
from ElectroWeakAnalysis.ZReco.dimuonsFilter_cfi import *
from ElectroWeakAnalysis.ZReco.dimuonsOneTrackFilter_cfi import *
dimuonsPath = cms.Path(dimuonsHLTFilter+goodMuonRecoForDimuon+dimuons+dimuonsGlobal+dimuonsOneStandAloneMuon+dimuonsFilter)
dimuonsOneTrackPath = cms.Path(dimuonsHLTFilter+goodMuonRecoForDimuon+dimuonsOneTrack+dimuonsOneTrackFilter)
dimuonsMCTruth = cms.Path(dimuonsHLTFilter+mcTruthForDimuons)

