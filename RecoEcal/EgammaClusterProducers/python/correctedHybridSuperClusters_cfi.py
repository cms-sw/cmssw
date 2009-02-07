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
        fBremVec = cms.vdouble(-0.0589, 0.1499, 0.9094, -0.0008905, 1.327),
        brLinearHighThr = cms.double(7.0),

        fEtEtaVec = cms.vdouble(1.0005, -0.5407, 0.9820,    0,
                                0,       0.5654, 11.78, 1.154,
                                0,            0,     0,      0,
                                0,            0,     0,      0,
                                8.0,     1.023,  -0.00181,
                                0, 0, 0)
    ),
    recHitProducer = cms.InputTag("ecalRecHit","EcalRecHitsEB")
)


