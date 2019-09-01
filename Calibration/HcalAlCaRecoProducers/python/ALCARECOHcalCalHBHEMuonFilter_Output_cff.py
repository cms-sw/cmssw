import FWCore.ParameterSet.Config as cms
 
# output block for alcastream HCAL HBHEMuon
# output module 
#  module alcastreamHcalHBHEMuonOutput = PoolOutputModule
OutALCARECOHcalCalHBHEMuonFilter_noDrop = cms.PSet(
    # use this in case of filter available
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalHBHEMuonFilter')
        ),
    outputCommands = cms.untracked.vstring(
        'keep *_hbhereco_*_*',
        'keep *_ecalRecHit_*_*',
        'keep *_offlineBeamSpot_*_*',
        'keep *_TriggerResults_*_*',
        'keep HcalNoiseSummary_hcalnoise_*_*',
        'keep recoTracks_globalMuons_*_*',
        'keep recoTrackExtras_globalMuons_*_*',
        'keep recoTracks_standAloneMuons_*_*',
        'keep recoTrackExtras_standAloneMuons_*_*',
        'keep recoTracks_generalTracks_*_*',
        'keep recoTrackExtras_generalTracks_*_*',
        'keep recoTracks_tevMuons_*_*',
        'keep recoTrackExtras_tevMuons_*_*',
        'keep *_offlinePrimaryVertices_*_*',
        'keep *_scalersRawToDigi_*_*',
        'keep *_muons_*_*',
        )
    )
 
import copy
OutALCARECOHcalCalHBHEMuonFilter=copy.deepcopy(OutALCARECOHcalCalHBHEMuonFilter_noDrop)
OutALCARECOHcalCalHBHEMuonFilter.outputCommands.insert(0,"drop *")
