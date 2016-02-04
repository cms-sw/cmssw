# The following comments couldn't be translated into the new config version:

#    "keep *_genParticles_*_*",

import FWCore.ParameterSet.Config as cms

from RecoJets.Configuration.RecoJets_EventContent_cff import *
from SimG4Core.Configuration.SimG4Core_EventContent_cff import *
softJetsEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_ctfWithMaterialTracks_*_*', 
        'keep *_globalMuons_*_*', 
        'keep edmTriggerResults_*_*_*', 
        'keep *_l1extraParticles_*_*')
)
#include "Configuration/EventContent/data/EventContent.cff"
#replace softJetsEventContent.outputCommands += AODSIMEventContent.outputCommands
softJetsEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('softJetsPath')
    )
)
softJetsEventContent.outputCommands.extend(RecoJetsAOD.outputCommands)
softJetsEventContent.outputCommands.extend(SimG4CoreAOD.outputCommands)

