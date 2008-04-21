import FWCore.ParameterSet.Config as cms

# AlCaReco for muon based alignment using tracks in the CSC overlap regions
OutALCARECOMuAlOverlaps = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOMuAlOverlaps')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_ALCARECOMuAlOverlaps_*_*')
)

