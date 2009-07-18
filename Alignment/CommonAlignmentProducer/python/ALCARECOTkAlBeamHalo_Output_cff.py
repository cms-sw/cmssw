import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using BeamHalo events
OutALCARECOTkAlBeamHalo_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlBeamHalo')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_ALCARECOTkAlBeamHalo_*_*', 
        'keep *_MEtoEDMConverter_*_*')
)

import copy
OutALCARECOTkAlBeamHalo = copy.deepcopy(OutALCARECOTkAlBeamHalo_noDrop)
OutALCARECOTkAlBeamHalo.outputCommands.insert(0, "drop *")
