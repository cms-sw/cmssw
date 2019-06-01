import FWCore.ParameterSet.Config as cms
from RecoEcal.EgammaClusterProducers.hybridSuperClusters_cfi import *
from RecoEcal.EgammaClusterProducers.multi5x5BasicClusters_cfi import *

from RecoEgamma.EgammaIsolationAlgos.electronTrackIsolations_cfi import trkIsol03CfgV1,trkIsol04CfgV1,trkIsol03CfgV2,trkIsol04CfgV2

#==============================================================================
# Producer of transient ecal driven gsf electrons
#==============================================================================

ecalDrivenGsfElectrons = cms.EDProducer("GsfElectronEcalDrivenProducer",

    # input collections
    previousGsfElectronsTag = cms.InputTag(""),
    pflowGsfElectronsTag = cms.InputTag(""),
    gsfElectronCoresTag = cms.InputTag("ecalDrivenGsfElectronCores"),
    pfMvaTag = cms.InputTag(""),

    gedElectronMode = cms.bool(False),

    # steering
    applyPreselection = cms.bool(False),
    ecalDrivenEcalEnergyFromClassBasedParameterization = cms.bool(True),
    ecalDrivenEcalErrorFromClassBasedParameterization = cms.bool(True),
    applyAmbResolution = cms.bool(False),
    useEcalRegression = cms.bool(False),
    useCombinationRegression = cms.bool(False),

    # preselection parameters (ecal driven electrons)
    preselection = cms.PSet(
        minSCEtBarrel = cms.double(4.0),
        minSCEtEndcaps = cms.double(4.0),
        maxDeltaEtaBarrel = cms.double(0.02),
        maxDeltaEtaEndcaps = cms.double(0.02),
        maxDeltaPhiBarrel = cms.double(0.15),
        maxDeltaPhiEndcaps = cms.double(0.15),
        maxHOverEBarrelCone = cms.double(0.15),
        maxHOverEEndcapsCone = cms.double(0.15),
        maxHOverEBarrelTower = cms.double(0.15),
        maxHOverEEndcapsTower = cms.double(0.15),
    ),

    # Ecal rec hits configuration
    recHitFlagsToBeExcludedBarrel = cleanedHybridSuperClusters.RecHitFlagToBeExcluded,
    recHitFlagsToBeExcludedEndcaps = multi5x5BasicClustersCleaned.RecHitFlagToBeExcluded,
    recHitSeverityToBeExcludedBarrel = cleanedHybridSuperClusters.RecHitSeverityToBeExcluded,
    recHitSeverityToBeExcludedEndcaps = cleanedHybridSuperClusters.RecHitSeverityToBeExcluded,

    # Isolation algos configuration
    trkIsol03Cfg = trkIsol03CfgV1,
    trkIsol04Cfg = trkIsol04CfgV1,
    trkIsolHEEP03Cfg = trkIsol03CfgV2,
    trkIsolHEEP04Cfg = trkIsol04CfgV2,

    # Iso values
    useIsolationValues = cms.bool(False),

    SoftElecMVAFilesString = cms.vstring("RecoEgamma/ElectronIdentification/data/TMVA_BDTSoftElectrons_9Dec2013.weights.xml"),
)

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
pp_on_AA_2018.toModify(ecalDrivenGsfElectrons.preselection, minSCEtBarrel = 15.0)
pp_on_AA_2018.toModify(ecalDrivenGsfElectrons.preselection, minSCEtEndcaps = 15.0)

#==============================================================================
# Final producer of persistent gsf electrons
#==============================================================================

gsfElectrons = cms.EDProducer("GsfElectronProducer",

    # input collections
    previousGsfElectronsTag = cms.InputTag("ecalDrivenGsfElectrons"),
    pflowGsfElectronsTag = cms.InputTag("pfElectronTranslator","pf"),
    gsfElectronCoresTag = cms.InputTag("gedGsfElectronCores"),
    pfMvaTag = cms.InputTag("pfElectronTranslator","pf"),

    gedElectronMode = cms.bool(False),

    # steering
    applyPreselection = cms.bool(True),
    ecalDrivenEcalEnergyFromClassBasedParameterization = cms.bool(True),
    ecalDrivenEcalErrorFromClassBasedParameterization = cms.bool(True),
    applyAmbResolution = cms.bool(True),
    useEcalRegression = cms.bool(False),
    useCombinationRegression = cms.bool(False),

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
        minMVA = cms.double(-0.1),
        minMvaByPassForIsolated = cms.double(-0.1),
    ),

    # preselection parameters (tracker driven only electrons)
    preselectionPflow = cms.PSet(
        minMVAPflow = cms.double(-0.1),
        minMvaByPassForIsolatedPflow = cms.double(-0.1),
    ),

    # Ecal rec hits configuration
    recHitFlagsToBeExcludedBarrel = cleanedHybridSuperClusters.RecHitFlagToBeExcluded,
    recHitFlagsToBeExcludedEndcaps = multi5x5BasicClustersCleaned.RecHitFlagToBeExcluded,
    recHitSeverityToBeExcludedBarrel = cleanedHybridSuperClusters.RecHitSeverityToBeExcluded,
    recHitSeverityToBeExcludedEndcaps = cleanedHybridSuperClusters.RecHitSeverityToBeExcluded,

    # Iso values
    useIsolationValues = cms.bool(True),
)

ecalDrivenGsfElectronsFromMultiCl = ecalDrivenGsfElectrons.clone(
  gsfElectronCoresTag = "ecalDrivenGsfElectronCoresFromMultiCl",
)
