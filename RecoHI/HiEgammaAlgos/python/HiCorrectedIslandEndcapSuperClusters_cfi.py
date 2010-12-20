import FWCore.ParameterSet.Config as cms

# Energy scale correction for Island Endcap SuperClusters
correctedIslandEndcapSuperClusters = cms.EDProducer("HiEgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string(''),
    sigmaElectronicNoise = cms.double(0.15),
    superClusterAlgo = cms.string('Island'),
    etThresh = cms.double(0.0),
    rawSuperClusterProducer = cms.InputTag("islandSuperClusters","islandEndcapSuperClusters"),
    applyEnergyCorrection = cms.bool(True),
    isl_fCorrPset = cms.PSet(
        fEtaVect = cms.vdouble(0.993,0,0.00546,1.165,-0.180844,+0.040312),
        fBremVect = cms.vdouble(-0.773799,2.73438,-1.07235,0.986821,-0.0101822,0.000306744,1.00595,-0.0495958,0.00451986,1.00595,-0.0495958,0.00451986),
        fBremThVect = cms.vdouble(1.2,1.2),
        fEtEtaVect = cms.vdouble(0.9497,0.006985,1.03754,-0.0142667,-0.0233993,0,0,0.908915,0.0137322,16.9602,-29.3093,19.8976,-5.92666,0.654571),
        brLinearLowThr = cms.double(0.0),
        brLinearHighThr = cms.double(0.0),
	minR9Barrel = cms.double(0.94),
	minR9Endcap = cms.double(0.95),
        maxR9 = cms.double(1.5),
    ),
    VerbosityLevel = cms.string('ERROR'),
    recHitProducer = cms.InputTag("ecalRecHit","EcalRecHitsEE")
)


