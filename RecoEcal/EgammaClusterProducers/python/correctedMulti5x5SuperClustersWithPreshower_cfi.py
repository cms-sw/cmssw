import FWCore.ParameterSet.Config as cms

# Energy scale correction for Multi5x5 Endcap SuperClusters
correctedMulti5x5SuperClustersWithPreshower = cms.EDProducer("EgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string(''),
    sigmaElectronicNoise = cms.double(0.15),
    superClusterAlgo = cms.string('Multi5x5'),
    etThresh = cms.double(0.0),
    rawSuperClusterProducer = cms.InputTag("multi5x5SuperClustersWithPreshower"),
    applyEnergyCorrection = cms.bool(True),
    energyCorrectorName = cms.string("EcalClusterEnergyCorrectionObjectSpecific"),
    modeEB = cms.int32(0),
    modeEE = cms.int32(0),
    applyCrackCorrection = cms.bool(True),
    crackCorrectorName = cms.string('EcalClusterCrackCorrection'),
    applyLocalContCorrection= cms.bool(False),
    # energy correction
    fix_fCorrPset = cms.PSet(
       brLinearLowThr = cms.double(0.9),
       
       fBremVec = cms.vdouble(-0.05228, 0.08738, 0.9508, 0.002677, 1.221),
       brLinearHighThr = cms.double(6.0),
       
       fEtEtaVec = cms.vdouble(1,
                               -0.4386,  -32.38, 
                                0.6372,   15.67, 
                               -0.0928,  -2.462, 
                                1.138,  20.93)
       
    ),
    VerbosityLevel = cms.string('ERROR'),
    recHitProducer = cms.InputTag("ecalRecHit","EcalRecHitsEE")
)


uncleanedOnlyCorrectedMulti5x5SuperClustersWithPreshower = correctedMulti5x5SuperClustersWithPreshower.clone(
    rawSuperClusterProducer = "uncleanedOnlyMulti5x5SuperClustersWithPreshower"
)
