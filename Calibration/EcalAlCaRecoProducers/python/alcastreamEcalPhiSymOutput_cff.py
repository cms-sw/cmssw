import FWCore.ParameterSet.Config as cms

# output block for alcastream EcalPhiSym
# output module 
#  module alcastreamEcalPhiSymOutput = PoolOutputModule
alcastreamEcalPhiSymOutput = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *', 
        'keep EcalRecHitsSorted__*_*')
)

