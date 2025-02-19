import FWCore.ParameterSet.Config as cms

# Energy scale correction for Hybrid SuperClusters
correctedDynamicHybridSuperClusters = cms.EDProducer("EgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string(''),
    sigmaElectronicNoise = cms.double(0.03),
    superClusterAlgo = cms.string('DynamicHybrid'),
    etThresh = cms.double(0.0),
    rawSuperClusterProducer = cms.InputTag("dynamicHybridSuperClusters"),
    applyEnergyCorrection = cms.bool(True),
    energyCorrectorName = cms.string("EcalClusterEnergyCorrection"),
    modeEB = cms.int32(3),
    modeEE = cms.int32(5),                                                       

    # energy correction
    dyn_fCorrPset = cms.PSet(
        brLinearLowThr = cms.double(0.0),
        fBremVec = cms.vdouble(-0.01762, 0.04224, 0.9793, 0.0008075, 1.774),
        brLinearHighThr = cms.double(12.0),
        fEtEtaVec = cms.vdouble(0.9929, -0.01751, 0.0, -4.636, 5.945, 
            737.9, 4.057, 8.0, 1.023, 0.0, 
            0.0, 1.0, 0.0)
    ),
    recHitProducer = cms.InputTag("ecalRecHit","EcalRecHitsEB")
                                                     
)


