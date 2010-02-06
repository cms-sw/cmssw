import FWCore.ParameterSet.Config as cms
# File: CSCHaloData_cfi.py
# Original Author: R. Remington, The University of Florida
# Description: Module to build CSCHaloData and put into the event
# Date: Oct. 15, 2009

CSCHaloData = cms.EDProducer("CSCHaloDataProducer",
                          
                             # Digi Level
                             L1MuGMTReadoutLabel = cms.InputTag("gtDigis"),
                             
                             # HLT
                             HLTResultLabel = cms.InputTag("TriggerResults::HLT"),
                             HLTBitLabel = cms.VInputTag(    cms.InputTag("HLT_CSCBeamHalo"),
                                                             cms.InputTag("HLT_CSCBeamHaloOverlapRing1"),
                                                             cms.InputTag("HLT_CSCBeamHaloOverlapRing2"),
                                                             cms.InputTag("HLT_CSCBeamHaloRing2or3")
                                                             ),
                             

                             # RecHit Level
                             CSCRecHitLabel = cms.InputTag("csc2DRecHits"),
                             
                             # Higher Level Reco
                             CSCSegmentLabel= cms.InputTag("cscSegments"),
                             CosmicMuonLabel= cms.InputTag("cosmicMuons"),
                             MuonLabel = cms.InputTag("muons"),
                             SALabel  =  cms.InputTag("standAloneMuons"),

                             # Parameters optimized from MC BeamHalo.  These parameters select tracks with parallel-to-beam trajectories pointing towards HB/EB
                             DetaParam = cms.double(0.05),
                             DphiParam = cms.double(1.00),
                             NormChi2Param = cms.double(8.),
                             InnerRMinParam = cms.double(140.),
                             OuterRMinParam = cms.double(140.),
                             InnerRMaxParam = cms.double(310.),
                             OuterRMaxParam = cms.double(310.)
                             
                             )


