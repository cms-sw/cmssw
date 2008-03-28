import FWCore.ParameterSet.Config as cms

# Energy scale correction for Fixed Matrix Endcap SuperClusters
correctedFixedMatrixSuperClustersWithPreshower = cms.EDFilter("EgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string(''),
    sigmaElectronicNoise = cms.double(0.15),
    superClusterAlgo = cms.string('FixedMatrix'),
    etThresh = cms.double(0.0),
    rawSuperClusterProducer = cms.InputTag("fixedMatrixSuperClustersWithPreshower"),
    applyEnergyCorrection = cms.bool(True),
    # energy correction
    fix_fCorrPset = cms.PSet(
        brLinearThr = cms.double(0.0),
        fBremVec = cms.vdouble(0.0, 0.0, 0.0),
        fEtEtaVec = cms.vdouble(0.0, 0.0, 0.0)
    ),
    VerbosityLevel = cms.string('ERROR'),
    recHitProducer = cms.InputTag("ecalRecHit","EcalRecHitsEE")
)


