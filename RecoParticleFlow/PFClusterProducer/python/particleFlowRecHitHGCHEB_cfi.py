import FWCore.ParameterSet.Config as cms

particleFlowRecHitHGCHEB = cms.EDProducer("PFRecHitProducer",
    useHitMap = cms.untracked.bool(True),
    navigator = cms.PSet(
        name = cms.string("PFRecHitHGCHENavigator"),
        topologySource = cms.string("HGCalHEScintillatorSensitive"),
    ),
    producers = cms.VPSet(
           cms.PSet(
             name = cms.string("PFHGCHEBRecHitCreator"),
             src  = cms.InputTag("HGCalRecHit:HGCHEBRecHits"),
             geometryInstance = cms.string("HGCalHEScintillatorSensitive"),
             qualityTests = cms.VPSet( 
                cms.PSet(
                  name = cms.string("PFRecHitQTestThresholdInMIPs"),
                  thresholdInMIPs = cms.double(1.01),
                  mipValueInkeV = cms.double(1498.4),
                  recHitEnergyIs_keV = cms.bool(False),
                  recHitEnergyMultiplier = cms.double(1.0)
                  ),
                )
           )          
    )          
)
