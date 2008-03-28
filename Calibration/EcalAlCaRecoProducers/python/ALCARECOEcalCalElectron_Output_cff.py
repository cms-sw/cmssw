# The following comments couldn't be translated into the new config version:

#        "keep  edmTriggerResults_*_*_*"

import FWCore.ParameterSet.Config as cms

# output block for alcastream Electron
OutALCARECOEcalCalElectron = cms.PSet(
    # put this if you have a filter
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOEcalCalElectron')
    ),
    outputCommands = cms.untracked.vstring('drop  *', 'keep  *_electronFilter_*_*', 'keep  *_alCaIsolatedElectrons_*_*', 'keep edmTriggerResults_TriggerResults__HLT')
)

