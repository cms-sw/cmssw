import FWCore.ParameterSet.Config as cms

from DQMOffline.Muon.muonKinVsEtaAnalyzer_cfi import *
from DQMOffline.Muon.muonRecoAnalyzer_cfi import *
from DQMOffline.Muon.muonEnergyDepositAnalyzer_cfi import *
from DQMOffline.Muon.segmentTrackAnalyzer_cfi import *
from DQMOffline.Muon.muonSeedsAnalyzer_cfi import * 

# Cloning the modules that need some changes...
muonCosmicSeedAnalyzer          = muonSeedsAnalyzer.clone()
muonCosmicGlbSegmentAnalyzer    = glbMuonSegmentAnalyzer.clone()
muonCosmicStaSegmentAnalyzer    = staMuonSegmentAnalyzer.clone()

muonCosmicSeedAnalyzer.SeedCollection          = cms.InputTag('CosmicMuonSeed')
muonCosmicGlbSegmentAnalyzer.MuTrackCollection = cms.InputTag('globalCosmicMuons')
muonCosmicStaSegmentAnalyzer.MuTrackCollection = cms.InputTag('cosmicMuons')

muonCosmicAnalyzer = cms.Sequence(muonEnergyDepositAnalyzer*
                                  muonCosmicSeedAnalyzer*
                                  muonCosmicGlbSegmentAnalyzer*
                                  muonCosmicStaSegmentAnalyzer*
                                  muonRecoAnalyzer*
                                  muonKinVsEtaAnalyzer)
                                  
muonSACosmicAnalyzer = cms.Sequence(muonEnergyDepositAnalyzer*
                                    muonSeedsAnalyzer*
                                    muonRecoAnalyzer*
                                    glbMuonSegmentAnalyzer*
                                    staMuonSegmentAnalyzer*
                                    muonKinVsEtaAnalyzer)

                                  
