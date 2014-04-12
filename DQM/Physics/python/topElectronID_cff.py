import FWCore.ParameterSet.Config as cms

## test for electronId
simpleEleId70cIso = cms.EDProducer(
    "EleIdCutBasedExtProducer",
    src = cms.InputTag("gedGsfElectrons"),
    reducedBarrelRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
    verticesCollection = cms.InputTag("offlineBeamSpot"),
    dataMagneticFieldSetUp = cms.bool(False),
    dcsTag = cms.InputTag("scalersRawToDigi"),                                          
    algorithm = cms.string('eIDCB'),
    electronIDType  = cms.string('robust'),
    electronQuality = cms.string('70cIso'),
    electronVersion = cms.string('V04'),
    ## 70% point modified with restricting cuts to physical values                                          
    robust70cIsoEleIDCutsV04 = cms.PSet(
        barrel =  cms.vdouble(2.5e-02, 1.0e-02, 3.0e-02, 4.0e-03, -1, -1, 9999., 9999., 9999., 9999., 9999., 9999., 
                              9999., 9999., 9999., 9999., 9999., 4.0e-02, 0.0, -9999., 9999., 9999., 0, -1, 0.02, 0.02, ),
        endcap =  cms.vdouble(2.5e-02, 3.0e-02, 2.0e-02, 5.0e-03, -1, -1, 9999., 9999., 9999., 9999., 9999., 9999., 
                              9999., 9999., 9999., 9999., 9999., 3.0e-02, 0.0, -9999., 9999., 9999., 0, -1, 0.02, 0.02, ),
    ),
)
