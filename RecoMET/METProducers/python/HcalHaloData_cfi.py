import FWCore.ParameterSet.Config as cms
# File: HcalHaloData_cfi.py
# Original Author: R. Remington, The University of Florida
# Description: Module to build HcalHaloData Object and put into the event
# Date: Oct. 15, 2009

HcalHaloData = cms.EDProducer("HcalHaloDataProducer",
                              # RecHit Level
                              EBRecHitLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
                              EERecHitLabel = cms.InputTag("ecalRecHit", "EcalRecHitsEE"),
                              HBHERecHitLabel = cms.InputTag("hbhereco"),
                              HORecHitLabel  = cms.InputTag("horeco"),
                              HFRecHitLabel = cms.InputTag("hfreco"),

                              caloTowerCollName = cms.InputTag('towerMaker'),
                              
                              HcalMinMatchingRadiusParam = cms.double(110.),
                              HcalMaxMatchingRadiusParam = cms.double(490.),
                              HBRecHitEnergyThresholdParam = cms.double(0.5),
                              HERecHitEnergyThresholdParam = cms.double(0.5),
                              SumHcalEnergyThresholdParam = cms.double(18),
                              NHitsHcalThresholdParam = cms.int32(4),
                              
                              )


