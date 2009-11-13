import FWCore.ParameterSet.Config as cms
# File: CSCHaloData_cfi.py
# Original Author: R. Remington, The University of Florida
# Description: Module to build CSCHaloData and put into the event
# Date: Oct. 15, 2009

CSCHaloData = cms.EDProducer("CSCHaloDataProducer",
                          
                             # Digi Level
                             L1MuGMTReadoutLabel = cms.InputTag("gtDigis"),
                             
                             # RecHit Level
                             CSCRecHitLabel = cms.InputTag("csc2DRecHits"),
                             
                             # Higher Level Reco
                             CSCSegmentLabel= cms.InputTag("cscSegments"),
                             CosmicMuonLabel= cms.InputTag("cosmicMuons"),
                             MuonLabel = cms.InputTag("muons"),
                             SALabel  =  cms.InputTag("standAloneMuons"),
                             )


