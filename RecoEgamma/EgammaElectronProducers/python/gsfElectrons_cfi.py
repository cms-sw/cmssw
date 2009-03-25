import FWCore.ParameterSet.Config as cms

#==============================================================================
# Producer of gsf electrons
#==============================================================================

gsfElectrons = cms.EDProducer("GsfElectronProducer",

    # input collections
    tracks = cms.InputTag("electronGsfTracks"),
    gsfElectronCores = cms.InputTag("gsfElectronCores"),
    ctfTracks = cms.InputTag("generalTracks"),
    hcalTowers = cms.InputTag("towerMaker"),
    reducedBarrelRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    
    # steering
    applyPreselection = cms.bool(True),
    applyEtaCorrection = cms.bool(False),
    applyAmbResolution = cms.bool(True),
    
    # preselection parameters
    minSCEtBarrel = cms.double(4.0),
    minSCEtEndcaps = cms.double(4.0),
    minEOverPBarrel = cms.double(0.0),
    maxEOverPBarrel = cms.double(10000.0),
    minEOverPEndcaps = cms.double(0.0),
    maxEOverPEndcaps = cms.double(10000.0),
    maxDeltaEtaBarrel = cms.double(0.02),
    maxDeltaEtaEndcaps = cms.double(0.02),
    maxDeltaPhiBarrel = cms.double(0.15),
    maxDeltaPhiEndcaps = cms.double(0.15),
    hOverEConeSize = cms.double(0.15),
    hOverEPtMin = cms.double(0.),
    maxHOverEDepth1Barrel = cms.double(0.1),
    maxHOverEDepth1Endcaps = cms.double(0.1),
    maxHOverEDepth2 = cms.double(0.1),
    maxSigmaIetaIetaBarrel = cms.double(9999.),
    maxSigmaIetaIetaEndcaps = cms.double(9999.),
    maxFbremBarrel = cms.double(9999.),
    maxFbremEndcaps = cms.double(9999.),
    isBarrel = cms.bool(False),
    isEndcaps = cms.bool(False),
    isFiducial = cms.bool(False),
    seedFromTEC = cms.bool(True),
    
    # nested parameter set for TransientInitialStateEstimator
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )

)


