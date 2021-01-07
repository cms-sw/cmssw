import FWCore.ParameterSet.Config as cms

# Energy scale correction for Hybrid SuperClusters
correctedHybridSuperClusters = cms.EDProducer("EgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string(''),
    sigmaElectronicNoise = cms.double(0.03),
    superClusterAlgo = cms.string('Hybrid'),
    etThresh = cms.double(0.0),
    rawSuperClusterProducer = cms.InputTag("hybridSuperClusters"),
    applyEnergyCorrection = cms.bool(True),
    energyCorrectorName = cms.string("EcalClusterEnergyCorrectionObjectSpecific"),
    modeEB = cms.int32(0),
    modeEE = cms.int32(0),                     
    applyCrackCorrection = cms.bool(True),
    crackCorrectorName = cms.string('EcalClusterCrackCorrection'),
    applyLocalContCorrection= cms.bool(True),
    # energy correction
    hyb_fCorrPset = cms.PSet(

        brLinearLowThr = cms.double(1.1),
        fBremVec = cms.vdouble(-0.04382, 0.1169, 0.9267, -0.0009413, 1.419),
        brLinearHighThr = cms.double(8.0),

        fEtEtaVec = cms.vdouble(0,
                                1.00121, -0.63672,      0,      0,
                                0,         0.5655,  6.457, 0.5081,
                                8.0,     1.023,  -0.00181)
    ),
    recHitProducer = cms.InputTag("ecalRecHit","EcalRecHitsEB")
)


uncleanedOnlyCorrectedHybridSuperClusters =correctedHybridSuperClusters.clone(
    rawSuperClusterProducer = "hybridSuperClusters:uncleanOnlyHybridSuperClusters"
)
