import FWCore.ParameterSet.Config as cms
# File: BeamHaloAnalyzer_cfi.py
# Original Author: R. Remington, The University of Florida
# Description: Module monitor histograms related to the BeamHaloId tools in the RecoMET package 
# Date: Nov. 15, 2009

AnalyzeBeamHalo = cms.EDAnalyzer("BeamHaloAnalyzer",
                                 L1MuGMTReadoutLabel = cms.InputTag("gtDigis"),
                                 CSCRecHitLabel = cms.InputTag("csc2DRecHits"),
                                 EBRecHitLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
                                 EERecHitLabel = cms.InputTag("ecalRecHit", "EcalRecHitsEE"),
                                 ESRecHitLabel = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
                                 HBHERecHitLabel = cms.InputTag("hbhereco"),
                                 HORecHitLabel  = cms.InputTag("horeco"),
                                 HFRecHitLabel = cms.InputTag("hfreco"),
                                 CSCSegmentLabel= cms.InputTag("cscSegments"),
                                 metLabel = cms.InputTag("met"),
                                 CaloTowerLabel = cms.InputTag("towerMaker"),
                                 CollisionMuonLabel = cms.InputTag("muons"),
                                 CollisionStandAloneMuonLabel  =  cms.InputTag("standAloneMuons"),
                                 BeamHaloMuonLabel = cms.InputTag("muonsBeamHaloEndCapsOnly"),
                                 CosmicStandAloneMuonLabel = cms.InputTag("muonsFromCosmics"),
                                 SuperClusterLabel = cms.InputTag("correctedHybridSuperClusters"),
                                 PhotonLabel = cms.InputTag("photons"),
                                 CSCHaloDataLabel = cms.InputTag("CSCHaloData"),
                                 EcalHaloDataLabel = cms.InputTag("EcalHaloData"),
                                 HcalHaloDataLabel = cms.InputTag("HcalHaloData"),
                                 GlobalHaloDataLabel = cms.InputTag("GlobalHaloData"),
                                 BeamHaloSummaryLabel = cms.InputTag("BeamHaloSummary"),
                                 OutputFile = cms.string(""),
                                 TextFile = cms.string(""),
                                 DumpMET = cms.double(20.),
                                 folderName = cms.string("JetMET/BeamHalo"),
                                 StandardDQM = cms.bool(True),
                                 # Muon-Segment matching
                                 MatchParameters = cms.PSet(
                                     CSCsegments = cms.InputTag("cscSegments"),
                                     DTradius = cms.double(0.01),
                                     DTsegments = cms.InputTag("dt4DSegments"),
                                     RPChits = cms.InputTag("rpcRecHits"),
                                     TightMatchDT = cms.bool(False),
                                     TightMatchCSC = cms.bool(True)
                                     ),

#                                 OutputFile = cms.string("BeamHaloIdDQM.root"),
#                                 folderName = cms.string("JetMET/BeamHalo/")
                                 )
