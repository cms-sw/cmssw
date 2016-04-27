import FWCore.ParameterSet.Config as cms
# File: EcalHaloData_cfi.py
# Original Author: R. Remington, The University of Florida
# Description: Module to build EcalHaloData Object and put into the event
# Date: Oct. 15, 2009

EcalHaloData= cms.EDProducer("EcalHaloDataProducer",
                             # RecHit Level
                             EBRecHitLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
                             EERecHitLabel = cms.InputTag("ecalRecHit", "EcalRecHitsEE"),
                             ESRecHitLabel = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
                             HBHERecHitLabel = cms.InputTag("hbhereco"),
                             # Higher Level Reco
                             SuperClusterLabel = cms.InputTag("correctedHybridSuperClusters"),
#                             SuperClusterLabel = cms.InputTag("cosmicSuperClusters","CosmicBarrelSuperClusters"),
                             PhotonLabel = cms.InputTag("photons"),
                             
                             EBRecHitEnergyThresholdParam = cms.double(0.3),
                             EERecHitEnergyThresholdParam = cms.double(0.3),
                             ESRecHitEnergyThresholdParam = cms.double(0.3),
                             SumEcalEnergyThresholdParam = cms.double(10.),
                             NHitsEcalThresholdParam = cms.int32(4),

                             # Shower Shape cut parameters (defaults need to be optimized)
                             RoundnessCutParam  = cms.double(0.41),
                             AngleCutParam      = cms.double(0.51),
                             
                             )


