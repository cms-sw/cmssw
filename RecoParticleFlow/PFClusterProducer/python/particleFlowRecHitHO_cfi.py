import FWCore.ParameterSet.Config as cms
particleFlowRecHitHO = cms.EDProducer("PFRecHitProducer",
    navigator = cms.PSet(
        name = cms.string("PFRecHitHCALNavigator")
    ),
    producers = cms.VPSet(
           cms.PSet(
             name = cms.string("PFHORecHitCreator"),
             src  = cms.InputTag("horeco",""),
             qualityTests = cms.VPSet(
                  cms.PSet(
                    name = cms.string("PFRecHitQTestHOThreshold"),
                    threshold_ring0 = cms.double(0.4),
                    threshold_ring12 = cms.double(1.0)
                  ),
#                  cms.PSet(
#                  name = cms.string("PFRecHitQTestThreshold"),
#                  threshold = cms.double(0.05) # new threshold for SiPM HO
#                  ),
                  cms.PSet(
                      name = cms.string("PFRecHitQTestHCALChannel"),
                      maxSeverities      = cms.vint32(11),
                      cleaningThresholds = cms.vdouble(0.0),
                      flags              = cms.vstring('Standard')
                  )
             )
           )
    )

)

#
# Need to change the quality tests for post LS1 running
#
from Configuration.StandardSequences.Eras import eras

def _modifyParticleFlowRecHitHOForPostLS1( object ) :
    """
    Customises PFRecHitProducer for post LS1 by lowering the
    HO threshold for SiPM
    """
    for prod in object.producers:
        prod.qualityTests = cms.VPSet(
            cms.PSet(
                name = cms.string("PFRecHitQTestThreshold"),
                threshold = cms.double(0.05) # new threshold for SiPM HO
            ),
            cms.PSet(
                name = cms.string("PFRecHitQTestHCALChannel"),
                maxSeverities      = cms.vint32(11),
                cleaningThresholds = cms.vdouble(0.0),
                flags              = cms.vstring('Standard')
            )
        )

eras.run2.toModify( particleFlowRecHitHO, func=_modifyParticleFlowRecHitHOForPostLS1 )
