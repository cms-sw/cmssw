import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL Min Bias
# output module 
#  module alcastreamHcalMinbiasOutput = PoolOutputModule

OutALCARECOHcalCalMinBias_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalMinBias')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_hltTriggerSummaryAOD_*_*',
        'keep *_TriggerResults_*_*',
        'keep HcalNoiseSummary_hcalnoise_*_*',
        'keep HBHERecHitsSorted_hbherecoMBNZS_*_*',
        'keep HORecHitsSorted_horecoMBNZS_*_*',
        'keep HFRecHitsSorted_hfrecoMBNZS_*_*',
        'keep HBHERecHitsSorted_hbherecoNoise_*_*',
        'keep HORecHitsSorted_horecoNoise_*_*',
        'keep HFRecHitsSorted_hfrecoNoise_*_*')
)

import copy
OutALCARECOHcalCalMinBias=copy.deepcopy(OutALCARECOHcalCalMinBias_noDrop)
OutALCARECOHcalCalMinBias.outputCommands.insert(0, "drop *")

## customizations for the pp_on_AA_2018 eras
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
OutALCARECOHcalCalMinBiasHI = copy.deepcopy(OutALCARECOHcalCalMinBias_noDrop)
OutALCARECOHcalCalMinBiasHI.outputCommands.insert(0, "drop *")
OutALCARECOHcalCalMinBiasHI.outputCommands.insert(6, "keep HFRecHitsSorted_hfreco_*_*")

#Specify to use HI output for the pp_on_AA_2018 eras
pp_on_AA_2018.toReplaceWith(OutALCARECOHcalCalMinBias,OutALCARECOHcalCalMinBiasHI)
