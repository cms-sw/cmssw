import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using BeamHalo events
OutALCARECOTkAlBeamHalo = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlBeamHalo')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_ALCARECOTkAlBeamHalo_*_*')
)

