import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using MuonIsolated events
OutALCARECOTkAlMuonIsolated_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlMuonIsolated')
    ),
    outputCommands = cms.untracked.vstring( 
        'keep *_ALCARECOTkAlMuonIsolated_*_*', 
        'keep *_MEtoEDMConverter_*_*')
)

import copy
OutALCARECOTkAlMuonIsolated = copy.deepcopy(OutALCARECOTkAlMuonIsolated_noDrop)
OutALCARECOTkAlMuonIsolated.outputCommands.insert(0, "drop *")
