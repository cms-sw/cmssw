import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL HBHEMuon
# output module 
#  module alcastreamHcalHBHEMuonOutput = PoolOutputModule
OutALCARECOHcalCalHBHEMuonFilter_noDrop = cms.PSet(
    # use this in case of filter available
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalHBHEMuonFilter')
        ),
    outputCommands = cms.untracked.vstring( 
        'keep *_hbhereco_*_*',
        'keep *_ecalRecHit_*_*',
        'keep *_offlineBeamSpot_*_*',
        'keep *_hltTriggerSummaryAOD_*_*',
        'keep *_TriggerResults_*_*',
        'keep *_generalTracks_*_*',
        'keep *_generalTracksExtra_*_*',
        'keep *_offlinePrimaryVertices_*_*',
        'keep *_globalMuons_*_*',
        'keep *_standAloneMuons_*_*',
        'keep *_tevMuons_*_*',
        'keep *_muons_*_*',
        )
    )

import copy
OutALCARECOHcalCalHBHEMuonFilter=copy.deepcopy(OutALCARECOHcalCalHBHEMuonFilter_noDrop)
OutALCARECOHcalCalHBHEMuonFilter.outputCommands.insert(0,"drop *")
