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
    "keep recoGsfElectrons_*_*_*",
    'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducer_*_*',
    'keep recoPhotons_gedPhotonsTmp_*_*',
    'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerGED_*_*',
    'keep recoElectronSeeds_ecalDrivenElectronSeeds_*_*',
    'keep recoTrackExtras_electronGsfTracks_*_*'
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
    "keep recoGsfElectrons_*_*_*",
    'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducer_*_*',
    'keep recoPhotons_gedPhotonsTmp_*_*',
    'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerGED_*_*',
    'keep recoElectronSeeds_ecalDrivenElectronSeeds_*_*',
     'keep recoTrackExtras_electronGsfTracks_*_*'
    )
    )

RecoHiEgammaAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'keep floatedmValueMap_*_*_*',
    'keep recoGsfElectrons_gedGsfElectronsTmp_*_*',
    'keep recoSuperClusters_correctedIslandBarrelSuperClusters_*_*',
    'keep recoSuperClusters_correctedIslandEndcapSuperClusters_*_*',
    'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducer_*_*',
    'keep recoPhotons_gedPhotonsTmp_*_*',
    'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerGED_*_*',
    'keep recoElectronSeeds_ecalDrivenElectronSeeds_*_*',
    'keep recoTrackExtras_electronGsfTracks_*_*'
    )
    )
