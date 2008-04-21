# The following comments couldn't be translated into the new config version:

#     "keep *_genParticles_*_*",

import FWCore.ParameterSet.Config as cms

from SimG4Core.Configuration.SimG4Core_EventContent_cff import *
diMuonEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_ctfWithMaterialTracks_*_*', 
        'keep *_globalMuons_*_*', 
        'keep edmTriggerResults_*_*_*', 
        'keep *_l1extraParticles_*_*')
)
#include "Configuration/EventContent/data/EventContent.cff"
#replace diMuonEventContent.outputCommands += AODSIMEventContent.outputCommands
diMuonEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('diMuonPath')
    )
)
diMuonEventContent.outputCommands.extend(SimG4CoreAOD.outputCommands)

