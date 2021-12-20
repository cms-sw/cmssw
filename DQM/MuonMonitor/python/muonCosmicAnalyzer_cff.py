import FWCore.ParameterSet.Config as cms


from RecoMuon.TrackingTools.MuonServiceProxy_cff import *                                                                                        
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer                                                                                         
dqmInfoMuons = DQMEDAnalyzer('DQMEventInfo',                                                                                                    
                             subSystemFolder = cms.untracked.string('Muons')                                                                    
)                                                                                                                                               


from DQMOffline.Muon.segmentTrackAnalyzer_cfi import *
from DQMOffline.Muon.muonSeedsAnalyzer_cfi import * 
from DQM.MuonMonitor.muonRecoAnalyzer_cfi import *

#muonCosmicGlbSegmentAnalyzer    = glbMuonSegmentAnalyzer.clone(
#MuTrackCollection = 'globalCosmicMuons'
#)
muonCosmicStaSegmentAnalyzer    = staMuonSegmentAnalyzer.clone(
MuTrackCollection = 'cosmicMuons'
)
muonCosmicSeedAnalyzer   = muonSeedsAnalyzer.clone(
SeedCollection   = 'CosmicMuonSeed'
)



muonCosmicAnalyzer = cms.Sequence(muonCosmicStaSegmentAnalyzer * muonCosmicSeedAnalyzer * muonRecoAnalyzer)
