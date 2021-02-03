import FWCore.ParameterSet.Config as cms

correctedHybridSuperClusters = cms.EDProducer("EgammaSCCorrectionMaker",
    applyCrackCorrection = cms.bool(True),
    applyEnergyCorrection = cms.bool(True),
    applyLocalContCorrection = cms.bool(True),
    corectedSuperClusterCollection = cms.string(''),
    crackCorrectorName = cms.string('EcalClusterCrackCorrection'),
    energyCorrectorName = cms.string('EcalClusterEnergyCorrectionObjectSpecific'),
    etThresh = cms.double(0.0),
    hyb_fCorrPset = cms.PSet(
        brLinearHighThr = cms.double(8.0),
        brLinearLowThr = cms.double(1.1),
        fBremVec = cms.vdouble(-0.04382, 0.1169, 0.9267, -0.0009413, 1.419),
        fEtEtaVec = cms.vdouble(
            0, 1.00121, -0.63672, 0, 0,
            0, 0.5655, 6.457, 0.5081, 8.0,
            1.023, -0.00181
        )
    ),
    localContCorrectorName = cms.string('EcalBasicClusterLocalContCorrection'),
    modeEB = cms.int32(0),
    modeEE = cms.int32(0),
    rawSuperClusterProducer = cms.InputTag("hybridSuperClusters"),
    recHitProducer = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    sigmaElectronicNoise = cms.double(0.03),
    superClusterAlgo = cms.string('Hybrid')
)
