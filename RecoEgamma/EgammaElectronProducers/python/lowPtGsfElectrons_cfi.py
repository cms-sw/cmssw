import FWCore.ParameterSet.Config as cms
from RecoEcal.EgammaClusterProducers.hybridSuperClusters_cfi import *
from RecoEcal.EgammaClusterProducers.multi5x5BasicClusters_cfi import *

from RecoEgamma.EgammaIsolationAlgos.electronTrackIsolations_cfi import trkIsol03CfgV1,trkIsol04CfgV1

lowPtGsfElectrons = cms.EDProducer(
    "LowPtGsfElectronProducer",
    
    # input collections
    previousGsfElectronsTag = cms.InputTag(""),
    pflowGsfElectronsTag = cms.InputTag(""),
    gsfElectronCoresTag = cms.InputTag("lowPtGsfElectronCores"),
    pfMvaTag = cms.InputTag(""),
    
    # steering
    applyPreselection = cms.bool(False), #@@ True
    ecalDrivenEcalEnergyFromClassBasedParameterization = cms.bool(True), #@@ False
    ecalDrivenEcalErrorFromClassBasedParameterization = cms.bool(True), #@@ False
    applyAmbResolution = cms.bool(False),
    useEcalRegression = cms.bool(False), #@@ True
    useCombinationRegression = cms.bool(False), #@@ True
    
    # preselection parameters (ecal driven electrons)
    preselection = cms.PSet(
        minSCEtBarrel = cms.double(4.0),
        minSCEtEndcaps = cms.double(4.0),
        maxDeltaEtaBarrel = cms.double(0.02),
        maxDeltaEtaEndcaps = cms.double(0.02),
        maxDeltaPhiBarrel = cms.double(0.15),
        maxDeltaPhiEndcaps = cms.double(0.15),
        maxHOverEBarrel = cms.double(0.15),
        maxHOverEEndcaps = cms.double(0.15),
        ),
    
    # Ecal rec hits configuration
    recHitFlagsToBeExcludedBarrel = cleanedHybridSuperClusters.RecHitFlagToBeExcluded,
    recHitFlagsToBeExcludedEndcaps = multi5x5BasicClustersCleaned.RecHitFlagToBeExcluded,
    recHitSeverityToBeExcludedBarrel = cleanedHybridSuperClusters.RecHitSeverityToBeExcluded,
    recHitSeverityToBeExcludedEndcaps = cleanedHybridSuperClusters.RecHitSeverityToBeExcluded,
    
    # Isolation algos configuration
    trkIsol03Cfg = trkIsol03CfgV1,
    trkIsol04Cfg = trkIsol04CfgV1,
    
    # regression. The labels are needed in all cases.
    ecalRefinedRegressionWeightLabels = cms.vstring(),
    #"gedelectron_EBCorrection_offline_v1"
    #"gedelectron_EECorrection_offline_v1"
    #"gedelectron_EBUncertainty_offline_v1"
    #"gedelectron_EEUncertainty_offline_v1"
    combinationRegressionWeightLabels = cms.vstring(),
    #"gedelectron_p4combination_offline"
    
    # Iso values
    useIsolationValues = cms.bool(False),
    
    )

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
pp_on_AA_2018.toModify(lowPtGsfElectrons, minSCEtBarrel = 15.0)
pp_on_AA_2018.toModify(lowPtGsfElectrons, minSCEtEndcaps = 15.0)
