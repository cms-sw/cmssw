import FWCore.ParameterSet.Config as cms

from DQMOffline.Muon.muonKinVsEtaAnalyzer_cfi import *
from DQMOffline.Muon.muonRecoAnalyzer_cfi import *
from DQMOffline.Muon.muonEnergyDepositAnalyzer_cfi import *
from DQMOffline.Muon.segmentTrackAnalyzer_cfi import *
from DQMOffline.Muon.muonSeedsAnalyzer_cfi import * 

# Cloning the modules that need some changes...
muonCosmicSeedAnalyzer = muonSeedsAnalyzer.clone(
    SeedCollection  = 'CosmicMuonSeed'
)
muonCosmicGlbSegmentAnalyzer    = glbMuonSegmentAnalyzer.clone(
    MuTrackCollection = 'globalCosmicMuons'
)
muonCosmicStaSegmentAnalyzer    = staMuonSegmentAnalyzer.clone(
    MuTrackCollection = 'cosmicMuons'
)

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

                                  
