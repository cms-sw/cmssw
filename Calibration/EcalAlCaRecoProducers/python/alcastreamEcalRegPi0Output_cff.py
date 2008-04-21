import FWCore.ParameterSet.Config as cms

# output block for alcastream EcalPi0
# output module 
alcastreamEcalPi0Output = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *', 
        'keep EcalRecHitsSorted_alCaPi0RegRecHits_*_*', 
        'keep EcalRecHitsSorted_alCaPi0BCRecHits_*_*')
)

