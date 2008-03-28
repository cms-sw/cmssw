import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL Min Bias
# output module 
#  module alcastreamHcalMinbiasOutput = PoolOutputModule
OutALCARECOHcalCalMinBias = cms.PSet(
    # use this to filter
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalMinBias')
    ),
    outputCommands = cms.untracked.vstring('drop *', 'keep HBHERecHitsSorted_MinProd_*_*', 'keep HORecHitsSorted_MinProd_*_*', 'keep HFRecHitsSorted_MinProd_*_*')
)

