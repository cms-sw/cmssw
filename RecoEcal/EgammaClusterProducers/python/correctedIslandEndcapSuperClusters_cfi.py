import FWCore.ParameterSet.Config as cms

# Energy scale correction for Island Endcap SuperClusters
correctedIslandEndcapSuperClusters = cms.EDFilter("EgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string(''),
    sigmaElectronicNoise = cms.double(0.15),
    superClusterAlgo = cms.string('Island'),
    etThresh = cms.double(0.0),
    rawSuperClusterProducer = cms.InputTag("islandSuperClusters","islandBarrelSuperClusters"),
    applyEnergyCorrection = cms.bool(True),
    isl_fCorrPset = cms.PSet(
        brLinearLowThr = cms.double(0.0),
        fBremVec = cms.vdouble(0.0),
        brLinearHighThr = cms.double(0.0),
        fEtEtaVec = cms.vdouble(0.0)
    ),
    VerbosityLevel = cms.string('ERROR'),
    recHitProducer = cms.InputTag("ecalRecHit","EcalRecHitsEB")
)


