import FWCore.ParameterSet.Config as cms

# output block for alcastream Electron
OutALCARECODtCalib_noDrop = cms.PSet(
    # put this if you have a filter
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECODtCalib')
    ),
    outputCommands = cms.untracked.vstring( 
        'keep *_dt4DSegments_*_*', 
        'keep *_dt4DSegmentsNoWire_*_*',
        'keep *_muonDTDigis_*_*', 
        'keep *_dttfDigis_*_*',
        'keep *_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep recoMuons_muons_*_*',
        'keep booledmValueMap_muid*_*_*')
)


import copy
OutALCARECODtCalib = copy.deepcopy(OutALCARECODtCalib_noDrop)
OutALCARECODtCalib.outputCommands.insert(0, "drop *")
