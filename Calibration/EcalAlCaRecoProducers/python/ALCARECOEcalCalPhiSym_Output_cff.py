import FWCore.ParameterSet.Config as cms

# output block for alcastream EcalPhiSym
# output module 
#  module alcastreamEcalPhiSymOutput = PoolOutputModule
OutALCARECOEcalCalPhiSym = cms.PSet(
    # put this if you have a filter
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOEcalCalPhiSym')
    ),
    outputCommands = cms.untracked.vstring('drop *', 'keep EcalRecHitsSorted_alCaPhiSymStream_*_*')
)

