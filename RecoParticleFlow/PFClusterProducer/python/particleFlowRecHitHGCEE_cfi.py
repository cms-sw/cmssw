import FWCore.ParameterSet.Config as cms

particleFlowRecHitHGCEE = cms.EDProducer("PFRecHitProducer",
    useHitMap = cms.untracked.bool(True),
    navigator = cms.PSet(
        name = cms.string("PFRecHitHGCEENavigator"),
        topologySource = cms.string("HGCalEESensitive"),
    ),
    producers = cms.VPSet(
           cms.PSet(
             name = cms.string("PFHGCEERecHitCreator"),
             src  = cms.InputTag("HGCalRecHit:HGCEERecHits"),
             geometryInstance = cms.string("HGCalEESensitive"),
             qualityTests = cms.VPSet( 
                cms.PSet(
                  name = cms.string("PFRecHitQTestThresholdInMIPs"),
                  thresholdInMIPs = cms.double(0.544),
                  mipValueInkeV = cms.double(55.1),
                  recHitEnergyIs_keV = cms.bool(False),
                  recHitEnergyMultiplier = cms.double(1.0)
                  ),
                )
           )          
    )          
)
