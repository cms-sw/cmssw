
import FWCore.ParameterSet.Config as cms

#HB HE HO rec hits

particleFlowRecHitHF = cms.EDProducer("PFRecHitProducer",

    navigator = cms.PSet(
        name = cms.string("PFRecHitHCALNavigator"),
        barrel = cms.PSet( ),
        endcap = cms.PSet( )
    ),
    producers = cms.VPSet(
           cms.PSet(
             name = cms.string("PFHFRecHitCreator"),
             src  = cms.InputTag("hfreco",""),
             EMDepthCorrection = cms.double(22.),
             HADDepthCorrection = cms.double(25.),
             thresh_HF = cms.double(0.4),
             ShortFibre_Cut = cms.double(60.),
             LongFibre_Fraction = cms.double(0.10),
             LongFibre_Cut = cms.double(120.),
             ShortFibre_Fraction = cms.double(0.01),
             HFCalib29 = cms.double(1.07),
             qualityTests = cms.VPSet(
                  cms.PSet(
                      name = cms.string("PFRecHitQTestHCALChannel"),
                      maxSeverities      = cms.vint32(11,9,9),
                      cleaningThresholds = cms.vdouble(0.0,120.,60.),
                      flags              = cms.vstring('Standard','HFLong','HFShort'),
                  ),
                  cms.PSet(
                      name = cms.string("PFRecHitQTestHCALThresholdVsDepth"),
                      cuts = cms.VPSet(
                             cms.PSet(
                                 depth = cms.int32(1),
                                 threshold = cms.double(1.2)),
                             cms.PSet(
                                 depth = cms.int32(2),
                                 threshold = cms.double(1.8))
                      )
                  )   
                      
          )
    )             
  )

)
