import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL Pedestal
# output module 
#  module alcastreamHcalMinbiasOutput = PoolOutputModule

OutALCARECOHcalCalPedestal_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalPedestal')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_gtDigisAlCaPedestal_*_*',
        'keep HBHERecHitsSorted_hbherecoPedestal_*_*',
        'keep HORecHitsSorted_horecoPedestal_*_*',
        'keep HFRecHitsSorted_hfrecoPedestal_*_*')
)

import copy
OutALCARECOHcalCalPedestal=copy.deepcopy(OutALCARECOHcalCalPedestal_noDrop)
OutALCARECOHcalCalPedestal.outputCommands.insert(0, "drop *")
