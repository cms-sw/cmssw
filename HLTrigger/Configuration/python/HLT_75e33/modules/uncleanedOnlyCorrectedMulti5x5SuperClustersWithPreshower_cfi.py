import FWCore.ParameterSet.Config as cms

uncleanedOnlyCorrectedMulti5x5SuperClustersWithPreshower = cms.EDProducer("EgammaSCCorrectionMaker",
    VerbosityLevel = cms.string('ERROR'),
    applyCrackCorrection = cms.bool(True),
    applyEnergyCorrection = cms.bool(True),
    applyLocalContCorrection = cms.bool(False),
    corectedSuperClusterCollection = cms.string(''),
    crackCorrectorName = cms.string('EcalClusterCrackCorrection'),
    energyCorrectorName = cms.string('EcalClusterEnergyCorrectionObjectSpecific'),
    etThresh = cms.double(0.0),
    fix_fCorrPset = cms.PSet(
        brLinearHighThr = cms.double(6.0),
        brLinearLowThr = cms.double(0.9),
        fBremVec = cms.vdouble(-0.05228, 0.08738, 0.9508, 0.002677, 1.221),
        fEtEtaVec = cms.vdouble(
            1, -0.4386, -32.38, 0.6372, 15.67,
            -0.0928, -2.462, 1.138, 20.93
        )
    ),
    localContCorrectorName = cms.string('EcalBasicClusterLocalContCorrection'),
    modeEB = cms.int32(0),
    modeEE = cms.int32(0),
    rawSuperClusterProducer = cms.InputTag("uncleanedOnlyMulti5x5SuperClustersWithPreshower"),
    recHitProducer = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    sigmaElectronicNoise = cms.double(0.15),
    superClusterAlgo = cms.string('Multi5x5')
)
