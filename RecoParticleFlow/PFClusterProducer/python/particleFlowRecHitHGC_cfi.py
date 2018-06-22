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
# Enabling PFRecHitQTestHGCalThresholdSNR will filter out of the PFRecHits, all the HGCRecHits with energy not exceeding
# 5 sigma noise                 
                cms.PSet(
                    name = cms.string("PFRecHitQTestHGCalThresholdSNR"),
                    thresholdSNR = cms.double(5.0),
                   ),
                )              
           ),
           cms.PSet(
             name = cms.string("PFHGCHEFRecHitCreator"),
             src  = cms.InputTag("HGCalRecHit:HGCHEFRecHits"),
             geometryInstance = cms.string("HGCalHESiliconSensitive"),
             qualityTests = cms.VPSet(
# Enabling PFRecHitQTestHGCalThresholdSNR will filter out of the PFRecHits, all the HGCRecHits with energy not exceeding
# 5 sigma noise                     
                cms.PSet(
                    name = cms.string("PFRecHitQTestHGCalThresholdSNR"),
                    thresholdSNR = cms.double(5.0),
                   ),                 
                )
           ),
           cms.PSet(
             name = cms.string("PFHGCHEBRecHitCreator"),
             src  = cms.InputTag("HGCalRecHit:HGCHEBRecHits"),
             geometryInstance = cms.string(""),
             qualityTests = cms.VPSet(
# Enabling PFRecHitQTestHGCalThresholdSNR will filter out of the PFRecHits, all the HGCRecHits with energy not exceeding
# 5 sigma noise                     
                cms.PSet(
                    name = cms.string("PFRecHitQTestHGCalThresholdSNR"),
                    thresholdSNR = cms.double(5.0),
                   ),                 
                )
           )
    )          
)

from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9
phase2_hgcalV9.toModify(particleFlowRecHitHGC,
    producers = {
        0 : dict(name = "PFHGCalEERecHitCreator"),
        1 : dict(name = "PFHGCalHSiRecHitCreator"),
        2 : dict(name = "PFHGCalHScRecHitCreator"),
    }
)
