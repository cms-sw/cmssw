import FWCore.ParameterSet.Config as cms

# AlCaReco for HO using ZMuMu events
OutALCARECOHcalCalZMuMu = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalZMuMu')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_ALCARECOHcalCalZMuMu_*_*', 
        'keep *_horeco_*_*')
)

