import FWCore.ParameterSet.Config as cms

particleFlowRecHitHGCHEF = cms.EDProducer("PFRecHitProducer",
    useHitMap = cms.untracked.bool(True),
    navigator = cms.PSet(
        name = cms.string("PFRecHitHGCHENavigator"),
        topologySource = cms.string("HGCalHESiliconSensitive"),
    ),
    producers = cms.VPSet(
           cms.PSet(
             name = cms.string("PFHGCHEFRecHitCreator"),
             src  = cms.InputTag("HGCalRecHit:HGCHEFRecHits"),
             geometryInstance = cms.string("HGCalHESiliconSensitive"),
             qualityTests = cms.VPSet( 
                cms.PSet(
                  name = cms.string("PFRecHitQTestThresholdInMIPs"),
                  thresholdInMIPs = cms.double(0.50),
                  mipValueInkeV = cms.double(85.0),
                  recHitEnergyIs_keV = cms.bool(False),
                  recHitEnergyMultiplier = cms.double(1.0)
                  ),
                )
           )          
    )          
)
