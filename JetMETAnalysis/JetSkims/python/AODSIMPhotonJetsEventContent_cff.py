import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from JetMETAnalysis.JetSkims.photonjets_EventContent_cff import *
AODSIMPhotonJetsEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
AODSIMPhotonJetsEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
AODSIMPhotonJetsEventContent.outputCommands.extend(photonjetsEventContent.outputCommands)

