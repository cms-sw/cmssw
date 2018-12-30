import FWCore.ParameterSet.Config as cms

from RecoEcal.EgammaClusterProducers.hybridSuperClusters_cfi import *
from RecoEcal.EgammaClusterProducers.multi5x5BasicClusters_cfi import *

from RecoEgamma.EgammaIsolationAlgos.electronTrackIsolations_cfi import trkIsol03CfgV1,trkIsol04CfgV1


gedGsfElectronsTmp = cms.EDProducer("GEDGsfElectronProducer",

    # input collections
    previousGsfElectronsTag = cms.InputTag(""),
    pflowGsfElectronsTag = cms.InputTag(""),
    gsfElectronCoresTag = cms.InputTag("gedGsfElectronCores"),
    pfMvaTag = cms.InputTag(""),

    # steering
    applyPreselection = cms.bool(True),
    ecalDrivenEcalEnergyFromClassBasedParameterization = cms.bool(False),
    ecalDrivenEcalErrorFromClassBasedParameterization = cms.bool(False),
    applyAmbResolution = cms.bool(False),
    useEcalRegression = cms.bool(True),
    useCombinationRegression = cms.bool(True),

    # preselection parameters (ecal driven electrons)
    minSCEtBarrel = cms.double(4.0),
    minSCEtEndcaps = cms.double(4.0),

    # Ecal rec hits configuration
    recHitFlagsToBeExcludedBarrel = cleanedHybridSuperClusters.RecHitFlagToBeExcluded,
    recHitFlagsToBeExcludedEndcaps = multi5x5BasicClustersCleaned.RecHitFlagToBeExcluded,
    recHitSeverityToBeExcludedBarrel = cleanedHybridSuperClusters.RecHitSeverityToBeExcluded,
    recHitSeverityToBeExcludedEndcaps = cleanedHybridSuperClusters.RecHitSeverityToBeExcluded,

    # Isolation algos configuration
    trkIsol03Cfg = trkIsol03CfgV1,
    trkIsol04Cfg = trkIsol04CfgV1,

    # regression. The labels are needed in all cases.
    ecalRefinedRegressionWeightLabels = cms.vstring("gedelectron_EBCorrection_offline_v1",
                                                    "gedelectron_EECorrection_offline_v1",
                                                    "gedelectron_EBUncertainty_offline_v1",
                                                    "gedelectron_EEUncertainty_offline_v1"),
    combinationRegressionWeightLabels = cms.vstring("gedelectron_p4combination_offline"),

    # Iso values
    useIsolationValues = cms.bool(False),
)


from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
pp_on_AA_2018.toModify(gedGsfElectronsTmp, minSCEtBarrel = 15.0)
pp_on_AA_2018.toModify(gedGsfElectronsTmp, minSCEtEndcaps = 15.0)
