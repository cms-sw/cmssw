import FWCore.ParameterSet.Config as cms

# the clients
from DQM.TrackingMonitor.ClientTrackEfficiencySTACosmicMuons_cff import *
from DQM.TrackingMonitor.ClientTrackEfficiencyTkTracks_cff import *
from DQMOffline.Muon.trackResidualsTest_cfi import *
from DQMOffline.Muon.muonRecoTest_cfi import *
from DQMOffline.Muon.muonTestSummary_cfi import *
from DQMOffline.Muon.muonTestSummaryCosmics_cfi import *
from DQMOffline.Muon.EfficencyPlotter_cfi import *
from DQMOffline.Muon.TriggerMatchEfficencyPlotter_cfi import *

muonSourcesQualityTests = cms.EDAnalyzer("QualityTester",
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQMOffline/Muon/data/QualityTests1.xml')
)
muonComp2RefQualityTests = cms.EDAnalyzer("QualityTester",
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQMOffline/Muon/data/Mu_Comp2RefChi2.xml')
)

muonComp2RefKolmoQualityTests = cms.EDAnalyzer("QualityTester",
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQMOffline/Muon/data/Mu_Comp2RefKolmogorov.xml')
)
muonClientsQualityTests = cms.EDAnalyzer("QualityTester",
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQMOffline/Muon/data/QualityTests2.xml')
)

cosmicMuonQualityTests = cms.Sequence(ClientTrackEfficiencyTkTracks*
                                      ClientTrackEfficiencySTACosmicMuons*
                                      muonSourcesQualityTests*
                                      muTrackResidualsTest*
                                      muRecoTest*
                                      muonClientsQualityTests*
                                      muonComp2RefQualityTests*
                                      muonComp2RefKolmoQualityTests*
                                      muonCosmicTestSummary)

muonQualityTests = cms.Sequence(muonSourcesQualityTests*
                                muTrackResidualsTest*
                                effPlotterLoose*
                                effPlotterMedium*
                                effPlotterTight*
                                muRecoTest*
                                muonClientsQualityTests*
                                muonComp2RefQualityTests*
                                muonComp2RefKolmoQualityTests*
                                muonTestSummary)

muonQualityTests_miniAOD = cms.Sequence(muonSourcesQualityTests*
                                        muTrackResidualsTest*
                                        effPlotterLooseMiniAOD*
                                        effPlotterMediumMiniAOD*
                                        effPlotterTightMiniAOD*
                                        muRecoTest*
                                        muonClientsQualityTests*
                                        muonComp2RefQualityTests*
                                        muonComp2RefKolmoQualityTests*
                                        muonTestSummary*
                                        triggerMatchEffPlotterTightMiniAOD)


