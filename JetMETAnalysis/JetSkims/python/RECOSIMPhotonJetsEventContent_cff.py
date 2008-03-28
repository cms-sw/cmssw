import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from JetMETAnalysis.JetSkims.photonjets_EventContent_cff import *
RECOSIMPhotonJetsEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RECOSIMPhotonJetsEventContent.outputCommands.extend(RECOSIMEventContent.outputCommands)
RECOSIMPhotonJetsEventContent.outputCommands.extend(photonjetsEventContent.outputCommands)

