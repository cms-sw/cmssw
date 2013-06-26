import FWCore.ParameterSet.Config as cms

# Energy scale correction for Fixed Matrix Endcap SuperClusters
correctedMulti5x5SuperClustersWithPreshower = cms.EDProducer("EgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string(''),
    sigmaElectronicNoise = cms.double(0.15),
    superClusterAlgo = cms.string('Multi5x5'),
    etThresh = cms.double(0.0),
    rawSuperClusterProducer = cms.InputTag("multi5x5SuperClustersWithPreshower"),
    applyEnergyCorrection = cms.bool(True),
    # energy correction
    fix_fCorrPset = cms.PSet(
        brLinearLowThr = cms.double(0.9),

        fBremVec = cms.vdouble(-0.0731, 0.133, 0.9262, -0.0008765, 1.08),
        brLinearHighThr = cms.double(6.0),

        fEtEtaVec = cms.vdouble(-0.0377, -915.9, 111.7, -135.8,
                                 0.523,   431.1, 114.2,  104.6,
                                 0,      -120.3, 203.6, -31.98,
                                 0,        1054, 182.1, 112.10,
                                 0,           0,     0,
                                 1,           1,     1),
        
        corrF = cms.vint32(0, 0, 1)
    ),
    VerbosityLevel = cms.string('ERROR'),
    recHitProducer = cms.InputTag("ecalRecHit","EcalRecHitsEE")
)


