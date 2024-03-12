import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL Dijets
# output module 
#  module alcastreamHcalDijetsOutput = PoolOutputModule
alcastreamHcalDijetsOutput = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *', 
                                           'keep HcalNoiseSummary_hcalnoise_*_*',
                                           'keep *_DiJProd_*_*')
)

# foo bar baz
# IaP7Zhggon3Zi
