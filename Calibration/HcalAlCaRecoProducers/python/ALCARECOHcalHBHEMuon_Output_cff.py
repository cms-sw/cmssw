import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL HBHEMuon
# output module 
#  module alcastreamHcalHBHEMuonOutput = PoolOutputModule
OutALCARECOHcalHBHEMuon_noDrop = cms.PSet(
    # use this in case of filter available
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalHBHEMuon')
    ),
    outputCommands = cms.untracked.vstring( 
        'keep *_DiJetsProd_*_*',
	'keep triggerTriggerEvent_*_*_*',
        'keep *_particleFlow_*_*',
        'keep recoPFBlocks_particleFlowBlock_*_*',
        'keep recoPFClusters_*_*_*',
        'keep *_fixedGridRhoFastjetAll_*_*',
        'keep recoTracks_generalTracks_*_*')
)

import copy
OutALCARECOHcalHBHEMuon=copy.deepcopy(OutALCARECOHcalHBHEMuon_noDrop)
OutALCARECOHcalHBHEMuon.outputCommands.insert(0,"drop *")
