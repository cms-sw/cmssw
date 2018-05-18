import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL Min Bias
# output module 
#  module alcastreamHcalMinbiasOutput = PoolOutputModule
alcastreamHcalMinbiasOutput = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *', 
        'keep HBHERecHitsSorted_MinProd_*_*', 
        'keep HcalNoiseSummary_hcalnoise_*_*',
        'keep HORecHitsSorted_MinProd_*_*', 
        'keep HFRecHitsSorted_MinProd_*_*')
)

