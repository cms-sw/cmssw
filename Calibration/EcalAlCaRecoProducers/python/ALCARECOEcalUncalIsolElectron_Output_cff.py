import FWCore.ParameterSet.Config as cms
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalIsolElectron_Output_cff import *
import copy

OutALCARECOEcalUncalElectron_noDrop=copy.deepcopy(OutALCARECOEcalCalElectron_noDrop)
OutALCARECOEcalUncalElectron_noDrop.outputCommands+=cms.untracked.vstring(
    'keep *_ecalDigis_*_*',
    'keep *EcalTriggerPrimitiveDigi*_ecalDigis_*_*',
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
    'keep reco*Clusters_pfElectronTranslator_*_*'
    )

OutALCARECOEcalUncalElectron.SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring('pathALCARECOEcalUncalZElectron', 'pathALCARECOEcalUncalZSCElectron', 'pathALCARECOEcalUncalWElectron')
    )


OutALCARECOEcalUncalWElectron=copy.deepcopy(OutALCARECOEcalUncalElectron)
OutALCARECOEcalUncalWElectron_noDrop=copy.deepcopy(OutALCARECOEcalUncalElectron_noDrop)

OutALCARECOEcalUncalWElectron.SelectEvents =  cms.untracked.PSet(
    SelectEvents = cms.vstring('pathALCARECOEcalUncalWElectron') )
OutALCARECOEcalUncalWElectron_noDrop.SelectEvents =  cms.untracked.PSet(
    SelectEvents = cms.vstring('pathALCARECOEcalUncalWElectron') )


OutALCARECOEcalUncalZElectron=copy.deepcopy(OutALCARECOEcalUncalElectron)
OutALCARECOEcalUncalZElectron_noDrop=copy.deepcopy(OutALCARECOEcalUncalElectron_noDrop)

OutALCARECOEcalUncalZElectron.SelectEvents =  cms.untracked.PSet(
    SelectEvents = cms.vstring('pathALCARECOEcalUncalZElectron', 'pathALCARECOEcalUncalZSCElectron')    )
OutALCARECOEcalUncalZElectron_noDrop.SelectEvents =  cms.untracked.PSet(
    SelectEvents = cms.vstring('pathALCARECOEcalUncalZElectron', 'pathALCARECOEcalUncalZSCElectron')    )
