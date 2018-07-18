import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL GammaJet
# output module
#  module alcastreamHcalGammaJetOutput = PoolOutputModule
OutALCARECOHcalCalGammaJet_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalGammaJet')
    ),
    outputCommands = cms.untracked.vstring(
            'keep HcalNoiseSummary_hcalnoise_*_*',
            'keep *_gedPhotons_*_*',
            'keep recoPhotonCores_gedPhotonCore_*_*',
            'keep recoSuperClusters_particleFlowEGamma_*_*',
            'keep recoCaloClusters_particleFlowEGamma_*_*',
            'keep *_particleBasedIsolation_gedPhotons_*',
            'keep *_reducedEcalRecHitsEB_*_*',
            'keep *_reducedEcalRecHitsEE_*_*',
            'keep *_reducedEcalRecHitsES_*_*',
            'keep *_PhotonIDProdGED_PhotonCutBasedIDLoose_*',
            'keep *_PhotonIDProdGED_PhotonCutBasedIDTight_*',
            'keep *_gedGsfElectrons_*_*',
            'keep *_allConversions_*_*',

            'keep recoPFJets_ak4PFJetsCHS_*_*',

            'keep *_hbhereco_*_*',
            'keep *_hfreco_*_*',
            'keep *_horeco_*_*',

            'keep recoTracks_generalTracks__*',
            'keep *_particleFlow_*_*',
            'keep *_particleFlowPtrs_*_*',
            'keep recoPFBlocks_particleFlowBlock_*_*',
            'keep recoPFClusters_particleFlowClusterHCAL_*_*',
            'keep recoPFClusters_particleFlowClusterHO_*_*',

            'keep *_fixedGridRhoFastjetAll_*_*',
            'keep *_pfMet_*_*',
            'keep *_offlinePrimaryVertices_*_*',
            'keep *_offlineBeamSpot_*_*',
            'keep *_TriggerResults_*_HLT',

            'keep *_genParticles_*_*',
            'keep *_ak4GenJets_*_*',
            'keep GenEventInfoProduct_generator_*_*'
 )
)

import copy
OutALCARECOHcalCalGammaJet=copy.deepcopy(OutALCARECOHcalCalGammaJet_noDrop)
OutALCARECOHcalCalGammaJet.outputCommands.insert(0, "drop *")
