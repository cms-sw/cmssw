import FWCore.ParameterSet.Config as cms

# Energy scale correction for Fixed Matrix Endcap SuperClusters
correctedFixedMatrixSuperClustersWithPreshower = cms.EDProducer("EgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string(''),
    sigmaElectronicNoise = cms.double(0.15),
    superClusterAlgo = cms.string('FixedMatrix'),
    etThresh = cms.double(0.0),
    rawSuperClusterProducer = cms.InputTag("fixedMatrixSuperClustersWithPreshower"),
    applyEnergyCorrection = cms.bool(True),
    # energy correction
    fix_fCorrPset = cms.PSet(
        brLinearLowThr = cms.double(0.9),
        fBremVec = cms.vdouble(-0.1234, 0.2347, 0.8831, 0.002377, 1.037),
        brLinearHighThr = cms.double(5.0),
        fEtEtaVec = cms.vdouble(1.002, -0.09255, 0.0, 0.0, -4.072, 
            67.93, -7.333, 0.0, 0.0, 0.0, 
            2.6),
        corrF = cms.vint32(0, 0, 1)
    ),
    VerbosityLevel = cms.string('ERROR'),
    recHitProducer = cms.InputTag("ecalRecHit","EcalRecHitsEE")
)


