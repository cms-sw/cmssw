import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using UpsilonMuMu events
OutALCARECOTkAlUpsilonMuMu = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlUpsilonMuMu')
    ),
    outputCommands = cms.untracked.vstring('drop *', 'keep *_ALCARECOTkAlUpsilonMuMu_*_*')
)

