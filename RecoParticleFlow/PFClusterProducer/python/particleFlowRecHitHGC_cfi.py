import FWCore.ParameterSet.Config as cms

particleFlowRecHitHGC = cms.EDProducer("PFRecHitProducer",
    navigator = cms.PSet(        
        name = cms.string("PFRecHitHGCNavigator"),
        hgcee = cms.PSet(
            name = cms.string("PFRecHitHGCEENavigator"),
            topologySource = cms.string("HGCalEESensitive")
            ),
        hgchef = cms.PSet(
            name = cms.string("PFRecHitHGCHENavigator"),
            topologySource = cms.string("HGCalHESiliconSensitive"),
            ),
        hgcheb = cms.PSet(
            name = cms.string("PFRecHitHGCHENavigator"),
            topologySource = cms.string("HGCalHEScintillatorSensitive"),
            )
    ),
    producers = cms.VPSet(
           cms.PSet(
             name = cms.string("PFHGCEERecHitCreator"),
             src  = cms.InputTag("HGCalRecHit:HGCEERecHits"),
             geometryInstance = cms.string("HGCalEESensitive"),
             qualityTests = cms.VPSet( 
                cms.PSet(
                  name = cms.string("PFRecHitQTestThresholdInThicknessNormalizedMIPs"),
                  thresholdInMIPs = cms.double(0.50),
                  mipValueInkeV = cms.double(27.55),
                  recHitEnergyIs_keV = cms.bool(False),
                  recHitEnergyMultiplier = cms.double(1.0),
                  geometryInstance = cms.string("HGCalEESensitive")
                  )
                )
           ),
           cms.PSet(
             name = cms.string("PFHGCHEFRecHitCreator"),
             src  = cms.InputTag("HGCalRecHit:HGCHEFRecHits"),
             geometryInstance = cms.string("HGCalHESiliconSensitive"),
             qualityTests = cms.VPSet( 
                cms.PSet(
                  name = cms.string("PFRecHitQTestThresholdInThicknessNormalizedMIPs"),
                  thresholdInMIPs = cms.double(0.50),
                  mipValueInkeV = cms.double(27.55),
                  recHitEnergyIs_keV = cms.bool(False),
                  recHitEnergyMultiplier = cms.double(1.0),
                  geometryInstance = cms.string("HGCalHESiliconSensitive")
                  )                
                )
           )
    )          
)
