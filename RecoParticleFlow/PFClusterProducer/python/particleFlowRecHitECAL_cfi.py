
import FWCore.ParameterSet.Config as cms
from RecoParticleFlow.PFClusterProducer.particleFlowZeroSuppressionECAL_cff import *

#until we are actually clustering across the EB/EE boundary
#it is faster to cluster EB and EE as separate

particleFlowRecHitECAL = cms.EDProducer("PFRecHitProducer",

    navigator = cms.PSet(
        name = cms.string("PFRecHitECALNavigator"),
        barrel = cms.PSet( ),
        endcap = cms.PSet( )
    ),
    producers = cms.VPSet(
           cms.PSet(
             name = cms.string("PFEBRecHitCreator"),
             src  = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
             srFlags = cms.InputTag("ecalDigis"),
             qualityTests = cms.VPSet(
                  cms.PSet(
                  name = cms.string("PFRecHitQTestECALMultiThreshold"),
                  thresholds = particle_flow_zero_suppression_ECAL.thresholds,
                  applySelectionsToAllCrystals = cms.bool(False)
                  ),
                  cms.PSet(
                  name = cms.string("PFRecHitQTestECAL"),
                  cleaningThreshold = cms.double(2.0),
                  timingCleaning = cms.bool(True),
                  topologicalCleaning = cms.bool(True),
                  skipTTRecoveredHits = cms.bool(True)
                  )
             )
           ),
          cms.PSet(
            name = cms.string("PFEERecHitCreator"),
            src  = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
            srFlags = cms.InputTag("ecalDigis"),
            qualityTests = cms.VPSet(
                 cms.PSet(
                 name = cms.string("PFRecHitQTestECALMultiThreshold"),
                 thresholds = particle_flow_zero_suppression_ECAL.thresholds,
                 applySelectionsToAllCrystals = cms.bool(False)
                 ),
                 cms.PSet(
                 name = cms.string("PFRecHitQTestECAL"),
                 cleaningThreshold = cms.double(2.0),
                 timingCleaning = cms.bool(True),
                 topologicalCleaning = cms.bool(True),
                 skipTTRecoveredHits = cms.bool(True)
                 )
            )
          )
    )
          
)
