import FWCore.ParameterSet.Config as cms

from RecoEcal.EgammaClusterProducers.hybridSuperClusters_cfi import *
from RecoEcal.EgammaClusterProducers.multi5x5BasicClusters_cfi import *

from RecoEgamma.EgammaIsolationAlgos.electronTrackIsolations_cfi import trkIsol03CfgV1,trkIsol04CfgV1


gedGsfElectronsTmp = cms.EDProducer("GEDGsfElectronProducer",

    # input collections
    previousGsfElectronsTag = cms.InputTag(""),
    pflowGsfElectronsTag = cms.InputTag(""),
    gsfElectronCoresTag = cms.InputTag("gedGsfElectronCores"),
    barrelRecHitCollectionTag = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    endcapRecHitCollectionTag = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    hcalTowers = cms.InputTag("towerMaker"),
    checkHcalStatus = cms.bool(True),
    pfMvaTag =  cms.InputTag(""),
    seedsTag = cms.InputTag("ecalDrivenElectronSeeds"),
    beamSpotTag = cms.InputTag("offlineBeamSpot"),
    gsfPfRecTracksTag = cms.InputTag("pfTrackElec"),
    egmPFCandidatesTag = cms.InputTag('particleFlowEGamma'),
    vtxTag = cms.InputTag('offlinePrimaryVertices'),
                                
    #output collections    
    outputEGMPFValueMap = cms.string(''),

    # backward compatibility mechanism for ctf tracks
    ctfTracksCheck = cms.bool(True),
    ctfTracksTag = cms.InputTag("generalTracks"),

    gedElectronMode = cms.bool(True),
    PreSelectMVA = cms.double(-0.1),	
    MaxElePtForOnlyMVA = cms.double(50.0),                                
    
    # steering
    useGsfPfRecTracks = cms.bool(True),
    applyPreselection = cms.bool(True),
    ecalDrivenEcalEnergyFromClassBasedParameterization = cms.bool(False),
    ecalDrivenEcalErrorFromClassBasedParameterization = cms.bool(False),
    pureTrackerDrivenEcalErrorFromSimpleParameterization = cms.bool(True),
    applyAmbResolution = cms.bool(False),
    ambSortingStrategy = cms.uint32(1),
    ambClustersOverlapStrategy = cms.uint32(1),
    addPflowElectrons = cms.bool(True), # this one should be transfered to the "core" level
    useEcalRegression = cms.bool(True),
    useCombinationRegression = cms.bool(True),                                    
    
    # preselection parameters (ecal driven electrons)
    minSCEtBarrel = cms.double(4.0),
    minSCEtEndcaps = cms.double(4.0),
    minEOverPBarrel = cms.double(0.0),
    maxEOverPBarrel = cms.double(999999999.),
    minEOverPEndcaps = cms.double(0.0),
    maxEOverPEndcaps = cms.double(999999999.),
    maxDeltaEtaBarrel = cms.double(0.02),
    maxDeltaEtaEndcaps = cms.double(0.02),
    maxDeltaPhiBarrel = cms.double(0.15),
    maxDeltaPhiEndcaps = cms.double(0.15),
    #useHcalTowers = cms.bool(True),
    #useHcalRecHits = cms.bool(False),
    hOverEConeSize = cms.double(0.15),
    hOverEPtMin = cms.double(0.),
    #maxHOverEDepth1Barrel = cms.double(0.1),
    #maxHOverEDepth1Endcaps = cms.double(0.1),
    #maxHOverEDepth2 = cms.double(0.1),
    maxHOverEBarrel = cms.double(0.15),
    maxHOverEEndcaps = cms.double(0.15),
    maxHBarrel = cms.double(0.0),
    maxHEndcaps = cms.double(0.0),
    maxSigmaIetaIetaBarrel = cms.double(999999999.),
    maxSigmaIetaIetaEndcaps = cms.double(999999999.),
    maxFbremBarrel = cms.double(999999999.),
    maxFbremEndcaps = cms.double(999999999.),
    isBarrel = cms.bool(False),
    isEndcaps = cms.bool(False),
    isFiducial = cms.bool(False),
    maxTIP = cms.double(999999999.),
    seedFromTEC = cms.bool(True),
    minMVA = cms.double(-0.4),
    minMvaByPassForIsolated = cms.double(-0.4),

    # preselection parameters (tracker driven only electrons)    
    minSCEtBarrelPflow = cms.double(0.0),
    minSCEtEndcapsPflow = cms.double(0.0),
    minEOverPBarrelPflow = cms.double(0.0),
    maxEOverPBarrelPflow = cms.double(999999999.),
    minEOverPEndcapsPflow = cms.double(0.0),
    maxEOverPEndcapsPflow = cms.double(999999999.),
    maxDeltaEtaBarrelPflow = cms.double(999999999.),
    maxDeltaEtaEndcapsPflow = cms.double(999999999.),
    maxDeltaPhiBarrelPflow = cms.double(999999999.),
    maxDeltaPhiEndcapsPflow = cms.double(999999999.),
    hOverEConeSizePflow = cms.double(0.15),
    hOverEPtMinPflow = cms.double(0.),
    #maxHOverEDepth1BarrelPflow = cms.double(999999999.),
    #maxHOverEDepth1EndcapsPflow = cms.double(999999999.),
    #maxHOverEDepth2Pflow = cms.double(999999999.),
    maxHOverEBarrelPflow = cms.double(999999999.),
    maxHOverEEndcapsPflow = cms.double(999999999.),
    maxHBarrelPflow = cms.double(0.0),
    maxHEndcapsPflow = cms.double(0.0),
    maxSigmaIetaIetaBarrelPflow = cms.double(999999999.),
    maxSigmaIetaIetaEndcapsPflow = cms.double(999999999.),
    maxFbremBarrelPflow = cms.double(999999999.),
    maxFbremEndcapsPflow = cms.double(999999999.),
    isBarrelPflow = cms.bool(False),
    isEndcapsPflow = cms.bool(False),
    isFiducialPflow = cms.bool(False),
    maxTIPPflow = cms.double(999999999.),
    minMVAPflow = cms.double(-0.4),
    minMvaByPassForIsolatedPflow = cms.double(-0.4),
    
    # Ecal rec hits configuration
    recHitFlagsToBeExcludedBarrel = cleanedHybridSuperClusters.RecHitFlagToBeExcluded,
    recHitFlagsToBeExcludedEndcaps = multi5x5BasicClustersCleaned.RecHitFlagToBeExcluded,
    recHitSeverityToBeExcludedBarrel = cleanedHybridSuperClusters.RecHitSeverityToBeExcluded,
    recHitSeverityToBeExcludedEndcaps = cleanedHybridSuperClusters.RecHitSeverityToBeExcluded,
    #severityLevelCut = cms.int32(4),

    # Isolation algos configuration
    trkIsol03Cfg = trkIsol03CfgV1,
    trkIsol04Cfg = trkIsol04CfgV1,
    intRadiusHcal = cms.double(0.15),
    etMinHcal = cms.double(0.0), 
    intRadiusEcalBarrel = cms.double(3.0), 
    intRadiusEcalEndcaps = cms.double(3.0), 
    jurassicWidth = cms.double(1.5), 
    etMinBarrel = cms.double(0.0),
    eMinBarrel = cms.double(0.095), 
    etMinEndcaps = cms.double(0.110), 
    eMinEndcaps = cms.double(0.0),  
    vetoClustered  = cms.bool(False),  
    useNumCrystals = cms.bool(True),  
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    
    # Corrections
    superClusterErrorFunction = cms.string("EcalClusterEnergyUncertaintyObjectSpecific"),
    crackCorrectionFunction = cms.string("EcalClusterCrackCorrection"),

   # regression. The labels are needed in all cases
   ecalRefinedRegressionWeightLabels = cms.vstring('gedelectron_EBCorrection_offline_v1',
                                                   'gedelectron_EECorrection_offline_v1',
                                                   'gedelectron_EBUncertainty_offline_v1',
                                                   'gedelectron_EEUncertainty_offline_v1'),
   combinationRegressionWeightLabels = cms.vstring('gedelectron_p4combination_offline'),
   
   ecalWeightsFromDB = cms.bool(True),
   # if not from DB. Otherwise, keep empty
   ecalRefinedRegressionWeightFiles = cms.vstring(),
   combinationWeightsFromDB = cms.bool(True),
   # if not from DB. Otherwise, keep empty
   combinationRegressionWeightFile = cms.vstring(),                              
 
   # Iso Values 
   useIsolationValues = cms.bool(False),
 SoftElecMVAFilesString = cms.vstring(
    "RecoEgamma/ElectronIdentification/data/TMVA_BDTSoftElectrons_7Feb2014.weights.xml"
                                ),
 ElecMVAFilesString = cms.vstring(
     "RecoEgamma/ElectronIdentification/data/TMVA_Category_BDTSimpleCat_10_17Feb2011.weights.xml",
     "RecoEgamma/ElectronIdentification/data/TMVA_Category_BDTSimpleCat_12_17Feb2011.weights.xml",
     "RecoEgamma/ElectronIdentification/data/TMVA_Category_BDTSimpleCat_20_17Feb2011.weights.xml",
     "RecoEgamma/ElectronIdentification/data/TMVA_Category_BDTSimpleCat_22_17Feb2011.weights.xml"
                                 ),
)

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
pp_on_AA_2018.toModify(gedGsfElectronsTmp, minSCEtBarrel = 15.0)
pp_on_AA_2018.toModify(gedGsfElectronsTmp, minSCEtEndcaps = 15.0)
