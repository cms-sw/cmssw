import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL Min Bias
# output module 
#  module alcastreamHcalMinbiasOutput = PoolOutputModule

OutALCARECOHcalCalMinBiasHI_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalMinBias')
    ),
    outputCommands = cms.untracked.vstring(
	'keep *_gtDigisAlCaMB_*_*',
        'keep HBHERecHitsSorted_hbhereco_*_*',
        'keep HORecHitsSorted_horeco_*_*',
        'keep HFRecHitsSorted_hfreco_*_*',
        'keep HFRecHitsSorted_hfrecoMBspecial_*_*',
        'keep HBHERecHitsSorted_hbherecoNoise_*_*',
        'keep HORecHitsSorted_horecoNoise_*_*',
        'keep HFRecHitsSorted_hfrecoNoise_*_*')
)

import copy
OutALCARECOHcalCalMinBiasHI=copy.deepcopy(OutALCARECOHcalCalMinBiasHI_noDrop)
OutALCARECOHcalCalMinBiasHI.outputCommands.insert(0, "drop *")
