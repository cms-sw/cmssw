import FWCore.ParameterSet.Config as cms

# output block for alcastream
OutALCARECODtCalibCosmics_noDrop = cms.PSet(
    # put this if you have a filter
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECODtCalibCosmics')
    ),
    outputCommands = cms.untracked.vstring( 
        'keep *_dt4DSegments_*_*', 
        'keep *_dt4DSegmentsNoWire_*_*',
        'keep *_muonDTDigis_*_*', 
        'keep *_dttfDigis_*_*',
        'keep *_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep recoTracks_cosmicMuons_*_*',
        'keep recoMuons_muons_*_*',
        'keep booledmValueMap_muid*_*_*')
)


import copy
OutALCARECODtCalibCosmics = copy.deepcopy(OutALCARECODtCalibCosmics_noDrop)
OutALCARECODtCalibCosmics.outputCommands.insert(0, "drop *")
