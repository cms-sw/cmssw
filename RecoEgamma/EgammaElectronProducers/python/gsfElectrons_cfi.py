
import FWCore.ParameterSet.Config as cms

#==============================================================================
# Producer of gsf electrons
#==============================================================================

gsfElectrons = cms.EDProducer("GsfElectronProducer",

    # input collections
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
    ambSortingStrategy = cms.uint32(1),
    ambClustersOverlapStrategy = cms.uint32(1),
    addPflowElectrons = cms.bool(True),
    
    # preselection parameters (ecal driven electrons)
    minSCEtBarrel = cms.double(4.0),
    minSCEtEndcaps = cms.double(4.0),
    minEOverPBarrel = cms.double(0.0),
    maxEOverPBarrel = cms.double(9999.),
    minEOverPEndcaps = cms.double(0.0),
    maxEOverPEndcaps = cms.double(9999.),
    maxDeltaEtaBarrel = cms.double(0.02),
    maxDeltaEtaEndcaps = cms.double(0.02),
    maxDeltaPhiBarrel = cms.double(0.15),
    maxDeltaPhiEndcaps = cms.double(0.15),
    #useHcalTowers = cms.bool(True),
    #useHcalRecHits = cms.bool(False),
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
    minMVA = cms.double(-9999.),
    maxTIP = cms.double(9999.),

    # preselection parameters (tracker driven only electrons)    
    minSCEtBarrelPflow = cms.double(0.0),
    minSCEtEndcapsPflow = cms.double(0.0),
    minEOverPBarrelPflow = cms.double(0.0),
    maxEOverPBarrelPflow = cms.double(9999.),
    minEOverPEndcapsPflow = cms.double(0.0),
    maxEOverPEndcapsPflow = cms.double(9999.),
    maxDeltaEtaBarrelPflow = cms.double(9999.),
    maxDeltaEtaEndcapsPflow = cms.double(9999.),
    maxDeltaPhiBarrelPflow = cms.double(9999.),
    maxDeltaPhiEndcapsPflow = cms.double(9999.),
    hOverEConeSizePflow = cms.double(0.15),
    hOverEPtMinPflow = cms.double(0.),
    maxHOverEDepth1BarrelPflow = cms.double(9999.),
    maxHOverEDepth1EndcapsPflow = cms.double(9999.),
    maxHOverEDepth2Pflow = cms.double(9999.),
    maxSigmaIetaIetaBarrelPflow = cms.double(9999.),
    maxSigmaIetaIetaEndcapsPflow = cms.double(9999.),
    maxFbremBarrelPflow = cms.double(9999.),
    maxFbremEndcapsPflow = cms.double(9999.),
    isBarrelPflow = cms.bool(False),
    isEndcapsPflow = cms.bool(False),
    isFiducialPflow = cms.bool(False),
    minMVAPflow = cms.double(-0.4),
    maxTIPPflow = cms.double(9999.),
    
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
    useNumCrystals = cms.bool(True),  
    
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    
    # Corrections
    superClusterErrorFunction = cms.string("EcalClusterEnergyUncertainty")

)


