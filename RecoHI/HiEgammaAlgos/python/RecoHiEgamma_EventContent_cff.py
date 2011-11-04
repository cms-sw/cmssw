import FWCore.ParameterSet.Config as cms

RecoHiEgammaFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'keep recoSuperClusters_*_*_*',
    'keep recoCaloClusters_*_*_*',
    'keep EcalRecHitsSorted_*_*_*',
    'keep floatedmValueMap_*_*_*',
    'keep recoPFCandidates_*_*_*',
    "drop recoPFClusters_*_*_*",
    "keep recoElectronSeeds_*_*_*",
    "keep recoGsfElectrons_*_*_*"    
    )
    )

RecoHiEgammaRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'keep recoSuperClusters_*_*_*',
    'keep recoCaloClusters_*_*_*',
    'keep EcalRecHitsSorted_*_*_*',
    'keep floatedmValueMap_*_*_*',  # isolation not created yet in RECO step, but in case it is later
    'keep recoPFCandidates_*_*_*',
    "drop recoPFClusters_*_*_*",
    "keep recoElectronSeeds_*_*_*",
    "keep recoGsfElectrons_*_*_*"    
    )
    )

RecoHiEgammaAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'keep floatedmValueMap_*_*_*'
    )
    )
