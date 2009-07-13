
import FWCore.ParameterSet.Config as cms

# output block for alcastream Electron
OutALCARECOEcalCalElectron_noDrop = cms.PSet(
    # put this if you have a filter
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOEcalCalElectron')
    ),
    outputCommands = cms.untracked.vstring( 
        'keep  *_electronFilter_*_*', 
        'keep  *_alCaIsolatedElectrons_*_*', 
        'keep edmTriggerResults_TriggerResults__HLT')
)


import copy
OutALCARECOEcalCalElectron=copy.deepcopy(OutALCARECOEcalCalElectron_noDrop)
OutALCARECOEcalCalElectron.outputCommands.insert(0, "drop *")
