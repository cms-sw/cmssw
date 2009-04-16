import FWCore.ParameterSet.Config as cms

#==============================================================================
# Producer of gsf electrons
#==============================================================================

gsfElectrons = cms.EDProducer("GsfElectronProducer",

    # input collections
    #tracks = cms.InputTag("electronGsfTracks"),
    gsfElectronCores = cms.InputTag("gsfElectronCores"),
    ctfTracks = cms.InputTag("generalTracks"),
    hcalTowers = cms.InputTag("towerMaker"),
    reducedBarrelRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    pfMVA =  cms.InputTag("pfElectronTranslator:pf"),
    
    # steering
    applyPreselection = cms.bool(True),
    applyEtaCorrection = cms.bool(False),
    applyAmbResolution = cms.bool(True),
    addPflowElectrons = cms.bool(True),
    
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
    # trackerDriven electrons
    minMVA = cms.double(-1.),
    
    # Isolation algos configuration
    intRadiusTk = cms.double(0.04), 
    ptMinTk = cms.double(0.7), 
    maxVtxDistTk = cms.double(0.2), 
    maxDrbTk = cms.double(9999.), 
    intRadiusHcal = cms.double(0.15),
    etMinHcal = cms.double(0.0), 
    intRadiusEcalBarrel = cms.double(3.0), 
    intRadiusEcalEndcaps = cms.double(3.0), 
    jurassicWidth = cms.double(1.5), 
    etMinBarrel = cms.double(0.0),
    eMinBarrel = cms.double(0.08), 
    etMinEndcaps = cms.double(0.1), 
    eMinEndcaps = cms.double(0.0),  
    vetoClustered  = cms.bool(False),  
    useNumCrystals = cms.bool(False),  
    
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )

)


