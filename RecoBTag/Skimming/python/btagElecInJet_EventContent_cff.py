import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
btagElecInJetEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
AODSIMbtagElecInJetEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RECOSIMbtagElecInJetEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
btagElecInJetEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('btagElecInJetPath')
    )
)
AODSIMbtagElecInJetEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
AODSIMbtagElecInJetEventContent.outputCommands.extend(btagElecInJetEventContent.outputCommands)
RECOSIMbtagElecInJetEventContent.outputCommands.extend(RECOSIMEventContent.outputCommands)
RECOSIMbtagElecInJetEventContent.outputCommands.extend(btagElecInJetEventContent.outputCommands)

