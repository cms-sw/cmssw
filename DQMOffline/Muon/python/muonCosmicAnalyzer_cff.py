import FWCore.ParameterSet.Config as cms

from DQMOffline.Muon.muonKinVsEtaAnalyzer_cfi import *
from DQMOffline.Muon.muonRecoAnalyzer_cfi import *
from DQMOffline.Muon.muonEnergyDepositAnalyzer_cfi import *
from DQMOffline.Muon.segmentTrackAnalyzer_cfi import *
from DQMOffline.Muon.muonSeedsAnalyzer_cfi import * 

# Cloning the modules that need some changes...
muonCosmicKinVsEtaAnayzer       = DQMOffline.Muon.muonKinVsEtaAnalyzer_cfi.clone()
muonCosmicRecoAnalyzer          = DQMOffline.Muon.muonRecoAnalyzer_cfi.clone()
muonCosmicEnergyDepositAnalyzer = DQMOffline.Muon.muonEnergyDepositAnalyzer_cfi.clone()
muonCosmicSeedAnalyzer          = DQMOffline.Muon.muonSeedsAnalyzer.clone()
muonCosmicGlbSegmentAnalyzer    = DQMOffline.Muon.glbMuonSegmentAnalyzer.clone()
muonCosmicStaSegmentAnalyzer    = DQMOffline.Muon.staMuonSegmentAnalyzer.clone()

muonCosmicSeedAnalyzer.SeedCollection          = cms.InputTag('CosmicMuonSeed')
muonCosmicGlbSegmentAnalyzer.MuTrackCollection = cms.InputTag('globalCosmicMuons')
muonCosmicStaSegmentAnalyzer.MuTrackCollection = cms.InputTag('cosmicMuons')

muonCosmicAnalyzer = cms.Sequence(muonCosmicEnergyDepositAnalyzer*
                                  muonCosmicSeedAnalyzer*
                                  muonCosmicGlbSegmentAnalyzer*
                                  muonCosmicStaSegmentAnalyzer*
                                  muonCosmicRecoAnalyzer*
                                  muonCosmicKinVsEtaAnayzer)
                                  
muonSACosmicAnalyzer = cms.Sequence(muonEnergyDepositAnalyzer*
                                    muonSeedsAnalyzer*
                                    muonRecoAnalyzer*
                                    glbMuonSegmentAnalyzer*
                                    staMuonSegmentAnalyzer*
                                    muonKinVsEtaAnalyzer)

                                  
