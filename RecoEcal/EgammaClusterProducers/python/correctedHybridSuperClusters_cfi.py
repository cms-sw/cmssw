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
        brLinearThr = cms.double(12.0),
        fBremVec = cms.vdouble(-0.01258, 0.03154, 0.9888, -0.0007973, 1.59),
        fEtEtaVec = cms.vdouble(1.0, -0.8206, 3.16, 0.8637, 44.88, 2.292, 1.023, 8.0)
    ),
    recHitProducer = cms.InputTag("ecalRecHit","EcalRecHitsEB")
)


