import FWCore.ParameterSet.Config as cms

# Energy scale correction for Multi5x5 Endcap SuperClusters
correctedMulti5x5SuperClustersWithPreshower = cms.EDFilter("EgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string(''),
    sigmaElectronicNoise = cms.double(0.15),
    superClusterAlgo = cms.string('Multi5x5'),
    etThresh = cms.double(0.0),
    rawSuperClusterProducer = cms.InputTag("multi5x5SuperClustersWithPreshower"),
    applyEnergyCorrection = cms.bool(True),
    # energy correction
    fix_fCorrPset = cms.PSet(
        brLinearLowThr = cms.double(0.6),
        fBremVec = cms.vdouble(-0.04163, 0.08552, 0.95048, -0.002308, 1.077),
        brLinearHighThr = cms.double(6.0),
        fEtEtaVec = cms.vdouble(0.9746, -6.512, 0, 0,
                                0.02771, 4.983, 0, 0,
                                -0.007288, -0.9446, 0, 0,
                                0, 0, 0, 1, 1)
    ),
    VerbosityLevel = cms.string('ERROR'),
    recHitProducer = cms.InputTag("ecalRecHit","EcalRecHitsEE")
)


