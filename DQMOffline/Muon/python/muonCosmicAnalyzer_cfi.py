import FWCore.ParameterSet.Config as cms

import DQMOffline.Muon.muonAnalyzer_cfi
# MuonAnalyzer
muonCosmicAnalyzer = DQMOffline.Muon.muonAnalyzer_cfi.muonAnalyzer.clone()
muonCosmicAnalyzer.MuonCollection = 'muons'
muonCosmicAnalyzer.SeedCollection = 'CosmicMuonSeed'
muonCosmicAnalyzer.GlobalMuTrackCollection = 'globalCosmicMuons'
muonCosmicAnalyzer.STAMuTrackCollection = 'cosmicMuons'


