import FWCore.ParameterSet.Config as cms

# output block for alcastream
OutALCARECODtCalibHI_noDrop = cms.PSet(
    # put this if you have a filter
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECODtCalibHI')
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
        'keep booledmValueMap_muid*_*_*',
        'keep *_hiSelectedVertex_*_*')
)


import copy
OutALCARECODtCalibHI = copy.deepcopy(OutALCARECODtCalibHI_noDrop)
OutALCARECODtCalibHI.outputCommands.insert(0, "drop *")
