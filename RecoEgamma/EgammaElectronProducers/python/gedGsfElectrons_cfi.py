import FWCore.ParameterSet.Config as cms

from RecoEcal.EgammaClusterProducers.hybridSuperClusters_cfi import *
from RecoEcal.EgammaClusterProducers.multi5x5BasicClusters_cfi import *

from RecoEgamma.EgammaIsolationAlgos.electronTrackIsolations_cfi import trkIsol03CfgV1,trkIsol04CfgV1,trkIsol03CfgV2,trkIsol04CfgV2


gedGsfElectronsTmp = cms.EDProducer("GEDGsfElectronProducer",

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

    # regression. The labels are needed in all cases.
    ecalRefinedRegressionWeightLabels = cms.vstring("gedelectron_EBCorrection_offline_v1",
                                                    "gedelectron_EECorrection_offline_v1",
                                                    "gedelectron_EBUncertainty_offline_v1",
                                                    "gedelectron_EEUncertainty_offline_v1"),
    combinationRegressionWeightLabels = cms.vstring("gedelectron_p4combination_offline"),
)


from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
pp_on_AA_2018.toModify(gedGsfElectronsTmp.preselection, minSCEtBarrel = 15.0)
pp_on_AA_2018.toModify(gedGsfElectronsTmp.preselection, minSCEtEndcaps = 15.0)

from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive
egamma_lowPt_exclusive.toModify(gedGsfElectronsTmp.preselection,
                           minSCEtBarrel = 1.0, 
                           minSCEtEndcaps = 1.0)
egamma_lowPt_exclusive.toModify(gedGsfElectronsTmp,
                           applyPreselection = False) 

