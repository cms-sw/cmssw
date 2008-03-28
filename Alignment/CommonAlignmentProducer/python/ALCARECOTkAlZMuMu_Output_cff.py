import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using ZMuMu events
OutALCARECOTkAlZMuMu = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlZMuMu')
    ),
    outputCommands = cms.untracked.vstring('drop *', 'keep *_ALCARECOTkAlZMuMu_*_*')
)

