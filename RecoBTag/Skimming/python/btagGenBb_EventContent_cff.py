import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
btagGenBbEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
AODSIMbtagGenBbEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RECOSIMbtagGenBbEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
btagGenBbEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('btagGenBbPath')
    )
)
AODSIMbtagGenBbEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
AODSIMbtagGenBbEventContent.outputCommands.extend(btagGenBbEventContent.outputCommands)
RECOSIMbtagGenBbEventContent.outputCommands.extend(RECOSIMEventContent.outputCommands)
RECOSIMbtagGenBbEventContent.outputCommands.extend(btagGenBbEventContent.outputCommands)

