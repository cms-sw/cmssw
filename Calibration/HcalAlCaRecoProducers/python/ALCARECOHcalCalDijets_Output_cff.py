import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL Dijets
# output module 
#  module alcastreamHcalDijetsOutput = PoolOutputModule
OutALCARECOHcalCalDijets_noDrop = cms.PSet(
    # use this in case of filter available
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalDijets')
    ),
    outputCommands = cms.untracked.vstring( 
        'keep *_DiJetsProd_*_*',
        'keep triggerTriggerEvent_*_*_*',
        'keep HcalNoiseSummary_hcalnoise_*_*',
        'keep *_particleFlow_*_*',
        'keep recoPFBlocks_particleFlowBlock_*_*',
        #'keep recoPFClusters_*_*_*',
        'keep recoPFClusters_particleFlowClusterECAL_*_*',
        'keep recoPFClusters_particleFlowClusterECALUncorrected_*_*',
        'keep recoPFClusters_particleFlowClusterHBHE_*_*',
        'keep recoPFClusters_particleFlowClusterHCAL_*_*',
        'keep recoPFClusters_particleFlowClusterHF_*_*',
        'keep recoPFClusters_particleFlowClusterHO_*_*',
        'keep recoPFClusters_particleFlowClusterPS_*_*',
        'keep *_fixedGridRhoFastjetAll_*_*',
        'keep recoTracks_generalTracks_*_*')
)

import copy
OutALCARECOHcalCalDijets=copy.deepcopy(OutALCARECOHcalCalDijets_noDrop)
OutALCARECOHcalCalDijets.outputCommands.insert(0,"drop *")
