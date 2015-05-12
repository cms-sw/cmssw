import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL HBHEMuon
# output module 
#  module alcastreamHcalHBHEMuonOutput = PoolOutputModule
alcastreamHcalHBHEMuonOutput = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *', 
                                           'keep *_HBHEMuonProd_*_*')
    )
