import FWCore.ParameterSet.Config as cms

# the clients
from DQM.TrackingMonitor.ClientTrackEfficiencySTACosmicMuons_cff import *
from DQM.TrackingMonitor.ClientTrackEfficiencyTkTracks_cff import *
from DQMOffline.Muon.trackResidualsTest_cfi import *
from DQMOffline.Muon.muonRecoTest_cfi import *
from DQMOffline.Muon.rpcClient_cfi import *
from DQMOffline.Muon.muonTestSummary_cff import *
muonSourcesQualityTests = cms.EDFilter("QualityTester",
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQMOffline/Muon/data/QualityTests1.xml')
)

muonClientsQualityTests = cms.EDFilter("QualityTester",
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQMOffline/Muon/data/QualityTests2.xml')
)

cosmicMuonQualityTests = cms.Sequence(ClientTrackEfficiencyTkTracks*ClientTrackEfficiencySTACosmicMuons*rpcClient*muonSourcesQualityTests*muTrackResidualsTest*muRecoTest*muonClientsQualityTests*muonTestSummary)

muonQualityTests = cms.Sequence(rpcClient*muonSourcesQualityTests*muTrackResidualsTest*muRecoTest*muonClientsQualityTests*muonTestSummary)

