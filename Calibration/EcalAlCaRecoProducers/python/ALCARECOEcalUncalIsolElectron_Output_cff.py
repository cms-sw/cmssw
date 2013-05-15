import FWCore.ParameterSet.Config as cms

# OutALCARECOEcalUncalElectron_specific = cms.untracked.vstring(
#     'keep *_TriggerResults_*_RECO',
#     #'keep *_TriggerResults_*_*',
#     #'keep recoGsfElectrons_gsfElectrons_*_RECO',
#     #'keep recoGsfElectronCores_gsfElectronCores_*_RECO',
#     'keep *EcalTriggerPrimitiveDigi*_ecalDigis_*_*',
#     'keep *_ecalGlobalUncalibRecHit_*_*',
#     'keep *_ecalPreshowerDigis_*_*',
#     'keep *_ecalDetIdToBeRecovered_*_*',
#     #this are recreated, so they are not needed at this step
#     'drop recoCaloClusters_*_*_*',
#     'drop recoSuperClusters_*_*_*',
#     'drop recoPreshowerCluster*_*_*_*',
#     'drop *EcalRecHit*_reducedEcalRecHitsES*_*_*',
#     # these are not recreated
#     'keep reco*Clusters_pfElectronTranslator_*_*'
#     )


from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalIsolElectron_Output_cff import *

import copy
OutALCARECOEcalUncalElectron_noDrop=copy.deepcopy(OutALCARECOEcalCalElectron_noDrop)
OutALCARECOEcalUncalElectron_noDrop.outputCommands+=cms.untracked.vstring(
    'keep *EcalTriggerPrimitiveDigi*_ecalDigis_*_*',
    'keep *_ecalGlobalUncalibRecHit_*_*',
    'keep *_ecalPreshowerDigis_*_*',
    'keep *_ecalDetIdToBeRecovered_*_*',
    'keep reco*Clusters_pfElectronTranslator_*_*'
    )

OutALCARECOEcalUncalElectron=copy.deepcopy(OutALCARECOEcalUncalElectron_noDrop)
OutALCARECOEcalUncalElectron.outputCommands.insert(0, "drop *")
OutALCARECOEcalUncalElectron.outputCommands += cms.untracked.vstring(
    'drop recoCaloClusters_*_*_*',
    'drop recoSuperClusters_*_*_*',
    'drop recoPreshowerCluster*_*_*_*',
    'drop *EcalRecHit*_reducedEcalRecHitsES*_*_*',
    'drop *_*Unclean*_*_*',
    'drop *_*unclean*_*_*',
    'drop *_*_*Unclean*_*',
    'drop *_*_*unclean*_*',
    )

OutALCARECOEcalUncalElectron.SelectEvents = cms.untracked.PSet(
    #SelectEvents = cms.vstring('pathALCARECOEcalUncalZElectron', 'pathALCARECOEcalUncalZSCElectron', 'pathALCARECOEcalUncalWElectron')
    SelectEvents = cms.vstring('pathALCARECOEcalUncalZElectron', 'pathALCARECOEcalUncalWElectron')
    )
