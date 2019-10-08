import FWCore.ParameterSet.Config as cms

from DQMOffline.Muon.segmentTrackAnalyzer_cfi import *


muonCosmicGlbSegmentAnalyzer    = glbMuonSegmentAnalyzer.clone()
muonCosmicStaSegmentAnalyzer    = staMuonSegmentAnalyzer.clone()

muonCosmicGlbSegmentAnalyzer.MuTrackCollection = cms.InputTag('globalCosmicMuons')
muonCosmicStaSegmentAnalyzer.MuTrackCollection = cms.InputTag('cosmicMuons')




muonCosmicAnalyzer = cms.Sequence(muonCosmicGlbSegmentAnalyzer*
                                  muonCosmicStaSegmentAnalyzer)

