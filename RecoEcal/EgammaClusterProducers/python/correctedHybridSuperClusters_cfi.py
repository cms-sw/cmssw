import FWCore.ParameterSet.Config as cms

# Energy scale correction for Hybrid SuperClusters
correctedHybridSuperClusters = cms.EDFilter("EgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string(''),
    sigmaElectronicNoise = cms.double(0.03),
    superClusterAlgo = cms.string('Hybrid'),
    etThresh = cms.double(0.0),
    rawSuperClusterProducer = cms.InputTag("hybridSuperClusters"),
    applyEnergyCorrection = cms.bool(True),
    VerbosityLevel = cms.string('ERROR'),
    # energy correction
    hyb_fCorrPset = cms.PSet(
        brLinearLowThr = cms.double(1.1),
        fBremVec = cms.vdouble(-0.05208, 0.1331, 0.9196, -0.0005735, 1.343),
        brLinearHighThr = cms.double(8.0),
        fEtEtaVec = cms.vdouble(1.0012, -0.5714, 0, 0,
                                0, 0.5549, 12.74, 1.0448,
                                0, 0, 0, 0,
                                8.0, 1.023, -0.00181, 0, 0)
    ),
    recHitProducer = cms.InputTag("ecalRecHit","EcalRecHitsEB")
)


