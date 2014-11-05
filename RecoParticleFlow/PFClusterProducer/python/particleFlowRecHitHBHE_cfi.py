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
#             offset_32 = cms.vdouble(1.6867*0.000843334685337,1.6867*0.00713160993668,1.6867*0.00323316695425,1.6867*0.00354196399691,1.6867*0.00243885183959),
#             offset_33 = cms.vdouble(1.6867*0.00169328273214,1.6867*0.0190872288871,1.6867*0.00624810899398,1.6867*0.0111491101175,1.6867*0.0128686529102),

             offset_32 = cms.vdouble(0.0237403833509,0.0422340301179,0.0262227205911,0.0190938508586,0.00955999701118),
             offset_33 = cms.vdouble(0.022447840761,0.0568700144634,0.029836140596,0.0347490310546,0.0322507475063),

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
                  )
                  

             )
           ),
           
    )

)

