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
        'keep *_DiJProd_*_*',
	'keep triggerTriggerEvent_*_*_*')
)

import copy
OutALCARECOHcalCalDijets=copy.deepcopy(OutALCARECOHcalCalDijets_noDrop)
OutALCARECOHcalCalDijets.outputCommands.insert(0,"drop *")
