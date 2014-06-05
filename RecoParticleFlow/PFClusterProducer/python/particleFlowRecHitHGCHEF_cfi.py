import FWCore.ParameterSet.Config as cms

particleFlowRecHitHGCHEF = cms.EDProducer("PFRecHitProducer",
    navigator = cms.PSet(
        name = cms.string("PFRecHitHGCHENavigator"),
        topologySource = cms.string("HGCalHESiliconSensitive"),
    ),
    producers = cms.VPSet(
           cms.PSet(
             name = cms.string("PFHGCHEBRecHitCreator"),
             src  = cms.InputTag("HGCalRecHit:HGCHEFRecHits"),
             geometryInstance = cms.string("HGCalHESiliconSensitive"),
             qualityTests = cms.VPSet( 
                cms.PSet(
                  name = cms.string("PFRecHitQTestThresholdInMIPs"),
                  thresholdInMIPs = cms.double(1.0),
                  mipValueInkeV = cms.double(85.0)
                  ),
                )
           )          
    )          
)
