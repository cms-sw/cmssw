import FWCore.ParameterSet.Config as cms

# output block for alcastream Electron
OutALCARECODtCalib_noDrop = cms.PSet(
    # put this if you have a filter
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('')
    ),
    outputCommands = cms.untracked.vstring( 
        'keep  *4DSegments*', 
        'keep  *muonDTDigis*', 
        'keep  *_dttfDigis_*_*',
        'keep *_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
	'keep *_MEtoEDMConverter_*_*')
)


import copy
OutALCARECODtCalib = copy.deepcopy(OutALCARECODtCalib_noDrop)
OutALCARECODtCalib.outputCommands.insert(0, "drop *")
