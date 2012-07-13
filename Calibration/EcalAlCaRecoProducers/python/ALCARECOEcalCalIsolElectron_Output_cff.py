import FWCore.ParameterSet.Config as cms

OutALCARECOEcalCalElectron = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOEcalCalElectron')
    ),
    outputCommands = cms.untracked.vstring(
    "drop *_*_*_*",
    "keep *_pfMet_*_*",
    "keep *_kt6PFJetsForRhoCorrection_rho_*",
    "keep *_kt6PFJets_rho_*",
    "keep *_offlinePrimaryVerticesWithBS_*_*",
    "keep *BeamSpot_offlineBeamSpot_*_*",
    "keep *_allConversions_*_*",
    "keep *_conversions_*_*",
    "keep *GsfTrack*_*_*_*",
    "keep *_generator_*_*",
    'keep *_addPileupInfo_*_*',
    'keep *_genParticles_*_*',
    'keep recoCaloClusters_*_*_*',
    'keep recoSuperClusters_*_*_*',
    'keep recoPreshowerCluster*_*_*_*',
    'drop reco*Clusters_hfEMClusters_*_RECO',
    'drop reco*Clusters_*Translator_*_RECO',
    'keep recoGsfElectron*_*_*_*',
    'keep *EcalRecHit*_alCaIsolatedElectrons_*_*',#+processName
    'keep *EcalRecHit*_reducedEcalRecHitsES_*_*',
    'drop *EcalRecHit*_ecalRecHit_*_*',
    'drop *EcalrecHit*_*ecalPreshowerRecHit*_*EcalRecHitsES*_*'
    )
)

