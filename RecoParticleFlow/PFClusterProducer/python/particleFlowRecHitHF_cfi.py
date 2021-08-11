import FWCore.ParameterSet.Config as cms

particleFlowRecHitHF = cms.EDProducer("PFRecHitProducer",

    navigator = cms.PSet(
        name = cms.string("PFRecHitHCALDenseIdNavigator"),
        hcalEnums = cms.vint32(4)
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
                      maxSeverities      = cms.vint32(11,9,9,9),
                      cleaningThresholds = cms.vdouble(0.0,120.,60.,0.),
                      flags              = cms.vstring('Standard','HFLong','HFShort','HFSignalAsymmetry'),
                  ),
                  cms.PSet(
                      name = cms.string("PFRecHitQTestHCALThresholdVsDepth"),
                      cuts = cms.VPSet(
                             cms.PSet(
                                 depth = cms.vint32(1,2),
                                 threshold = cms.vdouble(1.2,1.8),
                                 detectorEnum = cms.int32(4))
                      )
                  )

          )
    )
  )

)
