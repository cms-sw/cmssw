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
                             
                             # Chamber Level Trigger Primitive                             
                             ALCTDigiLabel = cms.InputTag("muonCSCDigis","MuonCSCALCTDigi"),

                             # RecHit Level
                             CSCRecHitLabel = cms.InputTag("csc2DRecHits"),
                             
                             # Higher Level Reco
                             CSCSegmentLabel= cms.InputTag("cscSegments"),
                             CosmicMuonLabel= cms.InputTag("cosmicMuons"),
                             MuonLabel = cms.InputTag("muons"),
                             SALabel  =  cms.InputTag("standAloneMuons"),

                             DetaParam = cms.double(0.1),
                             DphiParam = cms.double(1.00),
                             NormChi2Param = cms.double(8.),
                             InnerRMinParam = cms.double(0.),
                             OuterRMinParam = cms.double(0.),
                             InnerRMaxParam = cms.double(99999.),
                             OuterRMaxParam = cms.double(99999.),

                             MinOuterMomentumTheta = cms.double(.10),
                             MaxOuterMomentumTheta = cms.double(3.0),
                             MatchingDPhiThreshold = cms.double(0.18),
                             MatchingDEtaThreshold = cms.double(0.4),
                             MatchingDWireThreshold = cms.int32(5),
                             # The expected time of a collision recHit will be t = time_0 + time-of-flight
                             # A recHit more than +/- time_window from collision timing will be declared "out-of-time"                      
                             # recHit times are in [ns]
                             RecHitTime0 = cms.double(0.), 
                             RecHitTimeWindow = cms.double(25.),

                             # If this is Data, the expected collision bx will be 3 instead of 6
                             #ExpectedBX = cms.int32(3)   # if Data
                             ExpectedBX = cms.int32(6)   # if MC
                             )
