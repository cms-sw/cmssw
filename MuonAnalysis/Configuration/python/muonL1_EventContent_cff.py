import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
muonL1EventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
muonL1EventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('muonL1Path')
    )
)
RECOSIMmuonL1EventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
AODSIMmuonL1EventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RECOSIMmuonL1EventContent.outputCommands.extend(RECOSIMEventContent.outputCommands)
RECOSIMmuonL1EventContent.outputCommands.extend(muonL1EventContent.outputCommands)
AODSIMmuonL1EventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
AODSIMmuonL1EventContent.outputCommands.extend(muonL1EventContent.outputCommands)

