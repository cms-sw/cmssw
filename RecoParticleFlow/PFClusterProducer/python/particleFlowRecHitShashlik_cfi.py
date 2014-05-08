
import FWCore.ParameterSet.Config as cms

#until we are actually clustering across the EB/EE boundary
#it is faster to cluster EB and EE as separate

particleFlowRecHitShashlik = cms.EDProducer("PFRecHitProducer",
    navigator = cms.PSet(
        name = cms.string("PFRecHitShashlikNavigator"),
        barrel = cms.PSet( ),
        endcap = cms.PSet( )
    ),
    producers = cms.VPSet(
           cms.PSet(
             name = cms.string("PFEKRecHitCreator"),
             src  = cms.InputTag("shashlikRecHit"),
             qualityTests = cms.VPSet( )
           )          
    )          
)
