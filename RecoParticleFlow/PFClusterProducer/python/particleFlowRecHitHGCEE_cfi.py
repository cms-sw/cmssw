import FWCore.ParameterSet.Config as cms

particleFlowRecHitHGCEE = cms.EDProducer("PFRecHitProducer",
    useHitMap = cms.untracked.bool(True),
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
                  name = cms.string("PFRecHitQTestThresholdInMIPs"),
                  thresholdInMIPs = cms.double(0.544),
                  mipValueInkeV = cms.double(55.1),
                  recHitEnergyIs_keV = cms.bool(False),
                  recHitEnergyMultiplier = cms.double(1.0)
                  ),
                )
           ),
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
           ),
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
