import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL Min Bias
# output module 
#  module alcastreamHcalMinbiasOutput = PoolOutputModule

OutALCARECOHcalCalPedestal_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalPedestal')
    ),
    outputCommands = cms.untracked.vstring(
	'keep *_gtDigisAlCaMB_*_*',
        'keep HBHERecHitsSorted_hbherecoMB_*_*',
        'keep HORecHitsSorted_horecoMB_*_*',
        'keep HFRecHitsSorted_hfrecoMB_*_*',
        'keep HFRecHitsSorted_hfrecoMBspecial_*_*',
        'keep HBHERecHitsSorted_hbherecoNoise_*_*',
        'keep HORecHitsSorted_horecoNoise_*_*',
        'keep HFRecHitsSorted_hfrecoNoise_*_*')
)

import copy
OutALCARECOHcalCalPedestal=copy.deepcopy(OutALCARECOHcalCalPedestal_noDrop)
OutALCARECOHcalCalPedestal.outputCommands.insert(0, "drop *")
