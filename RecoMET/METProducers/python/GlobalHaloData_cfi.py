import FWCore.ParameterSet.Config as cms
# File: GlobalHaloData_cfi.py
# Original Author: R. Remington, The University of Florida
# Description: Module to build GlobalHaloData Object and put into the event
# Date: Oct. 15, 2009

GlobalHaloData = cms.EDProducer("GlobalHaloDataProducer",                         
                                # Higher Level Reco
                                metLabel = cms.InputTag("caloMet"),
                                calotowerLabel = cms.InputTag("towerMaker"),
                                CSCSegmentLabel = cms.InputTag("CSCSegments"),
                                CSCRecHitLabel = cms.InputTag("csc2DRecHits"),
                                
                                EcalMinMatchingRadiusParam = cms.double(110.),
                                EcalMaxMatchingRadiusParam  = cms.double(330.),
                                
                                HcalMinMatchingRadiusParam = cms.double(110.),
                                HcalMaxMatchingRadiusParam = cms.double(490.),

                                CSCHaloDataLabel = cms.InputTag("CSCHaloData"),
                                EcalHaloDataLabel = cms.InputTag("EcalHaloData"),
                                HcalHaloDataLabel = cms.InputTag("HcalHaloData"),

                                CaloTowerEtThresholdParam = cms.double(0.3)
                                                                
                                )


