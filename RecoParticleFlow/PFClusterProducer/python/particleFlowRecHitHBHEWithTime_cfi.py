import FWCore.ParameterSet.Config as cms

from particleFlowClusterECALTimeResolutionParameters_cfi import _timeResolutionHCAL


particleFlowRecHitHBHEWithTime = cms.EDProducer("PFRecHitProducer",
    navigator = cms.PSet(
        name = cms.string("PFRecHitHCALNavigatorWithTime"),
        noiseLevel = cms.double(0.0),   
        noiseTerm  = cms.double(0.0),
        constantTerm = cms.double(1.5),
        sigmaCut = cms.double(1),
        timeResolutionCalc = _timeResolutionHCAL
    ),
    producers = cms.VPSet(
           cms.PSet(
             name = cms.string("PFHBHERecHitCreatorPulses"),
             src  = cms.InputTag("hbhereco",""),
             qualityTests = cms.VPSet(
                  cms.PSet(
                  name = cms.string("PFRecHitQTestThreshold"),
                  threshold = cms.double(0.25)
                  ),
                  cms.PSet(
                      name = cms.string("PFRecHitQTestHCALChannel"),
                      maxSeverities      = cms.vint32(11),
                      cleaningThresholds = cms.vdouble(0.0),
                      flags              = cms.vstring('Standard')
                  )
                  

             )
           ),
           
    )

)

