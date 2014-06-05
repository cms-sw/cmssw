import FWCore.ParameterSet.Config as cms

particleFlowRecHitHGCHEB = cms.EDProducer("PFRecHitProducer",
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
                  thresholdInMIPs = cms.double(1.0),
                  mipValueInkeV = cms.double(1498.4)
                  ),
                )
           )          
    )          
)
