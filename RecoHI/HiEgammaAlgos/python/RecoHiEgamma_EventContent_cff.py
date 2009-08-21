import FWCore.ParameterSet.Config as cms

RecoHiEgammaFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'keep recoSuperClusters_*_*_*',
    'keep EcalRecHitsSorted_*_*_*',
    'keep floatedmValueMap_*_*_*'
    )
    )

RecoHiEgammaRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'keep recoSuperClusters_*_*_*',
    'keep EcalRecHitsSorted_*_*_*',
    'keep floatedmValueMap_*_*_*'  # isolation not created yet in RECO step, but in case it is later
    )
    )

RecoHiEgammaAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'keep floatedmValueMap_*_*_*'
    )
    )
