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
        brLinearLowThr = cms.double(0.7),
        fBremVec = cms.vdouble(-0.01217, 0.031, 0.9887, -0.0003776, 1.598),
        brLinearHighThr = cms.double(8.0),
        fEtEtaVec = cms.vdouble(1.001, -0.8654, 3.131, 0.0, 0.735, 
            20.72, 1.169, 8.0, 1.023, -0.00181, 
            0.0, 1.0, 0.0)
    ),
    recHitProducer = cms.InputTag("ecalRecHit","EcalRecHitsEB")
)


