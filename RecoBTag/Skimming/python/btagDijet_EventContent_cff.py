import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
btagDijetEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
AODSIMbtagDijetEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RECOSIMbtagDijetEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
btagDijetEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('btagDijetPath')
    )
)
AODSIMbtagDijetEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
AODSIMbtagDijetEventContent.outputCommands.extend(btagDijetEventContent.outputCommands)
RECOSIMbtagDijetEventContent.outputCommands.extend(RECOSIMEventContent.outputCommands)
RECOSIMbtagDijetEventContent.outputCommands.extend(btagDijetEventContent.outputCommands)

