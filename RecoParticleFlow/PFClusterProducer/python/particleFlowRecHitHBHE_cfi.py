import FWCore.ParameterSet.Config as cms
#from RecoParticleFlow.PFClusterProducer.particleFlowCaloResolution_cfi import _timeResolutionHCALMaxSample




particleFlowRecHitHBHE = cms.EDProducer("PFRecHitProducer",
    navigator = cms.PSet(
            name = cms.string("PFRecHitHCALNavigator"),
            #sigmaCut = cms.double(5.0),
            #timeResolutionCalc = _timeResolutionHCALMaxSample
    ),
    producers = cms.VPSet(
           cms.PSet(
             name = cms.string("PFHBHERecHitCreator"),
             src  = cms.InputTag("hbhereco",""),
             vertexSrc  = cms.InputTag("offlinePrimaryVertices"),

             offset_before_last = cms.vdouble(0.020,0.048,0.038,0.024,0.017),
             offset_last = cms.vdouble(0.013,0.069,0.043,0.040,0.037),

             qualityTests = cms.VPSet(
                  cms.PSet(
                  name = cms.string("PFRecHitQTestThreshold"),
                  threshold = cms.double(0.4)
                  ),
                  cms.PSet(
                      name = cms.string("PFRecHitQTestHCALChannel"),
                      maxSeverities      = cms.vint32(11),
                      cleaningThresholds = cms.vdouble(0.0),
                      flags              = cms.vstring('Standard')
                  ),
                  cms.PSet(
                      name = cms.string("PFRecHitQTestHCALTimeVsDepth"),
                      cuts = cms.VPSet(
                             cms.PSet( 
                                 depth = cms.int32(1),
                                 minTime = cms.double(-7.6),
                                 maxTime = cms.double(17.4),
                                 threshold = cms.double(0.0),
                             ),
                             cms.PSet( 
                                 depth = cms.int32(2),
                                 minTime = cms.double(-7.6),
                                 maxTime = cms.double(17.4),
                                 threshold = cms.double(0.0),
                             ),
                             cms.PSet( 
                                 depth = cms.int32(3),
                                 minTime = cms.double(-7.6),
                                 maxTime = cms.double(17.4),
                                 threshold = cms.double(0.0),
                             ),
                             cms.PSet( 
                                 depth = cms.int32(4),
                                 minTime = cms.double(-7.6),
                                 maxTime = cms.double(17.4),
                                 threshold = cms.double(0.0),
                             ),
                             cms.PSet( 
                                 depth = cms.int32(5),
                                 minTime = cms.double(-7.6),
                                 maxTime = cms.double(17.4),
                                 threshold = cms.double(0.0),
                             ),

                      )
                  ) 

             )
           ),
           
    )

)

