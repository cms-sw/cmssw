import FWCore.ParameterSet.Config as cms


sandboxOutputCommands = cms.untracked.vstring( [
        'drop *_TriggerResults_*_RECO',
#        'keep *_TriggerResults_*_*',
#        'keep recoGsfElectrons_gsfElectrons_*_RECO',
#        'keep recoGsfElectronCores_gsfElectronCores_*_RECO',
        'keep *EcalTriggerPrimitiveDigi*_ecalDigis_*_*',
        'keep *_ecalGlobalUncalibRecHit_*_*',
        'keep *_ecalPreshowerDigis_*_*',
        'keep *_ecalDetIdToBeRecovered_*_*',
# this are recreated, so they are not needed at this step
        'drop recoCaloClusters_*_*_*',
        'drop recoSuperClusters_*_*_*',
        'drop recoPreshowerCluster*_*_*_*',
        'drop *EcalRecHit*_reducedEcalRecHitsES*_*_*',
        # these are not recreated
        'keep reco*Clusters_pfElectronTranslator_*_*'
     ] )
