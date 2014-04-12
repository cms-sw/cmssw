import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
btagMuonInJetEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
AODSIMbtagMuonInJetEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RECOSIMbtagMuonInJetEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
btagMuonInJetEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('btagMuonInJetPath')
    )
)
AODSIMbtagMuonInJetEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
AODSIMbtagMuonInJetEventContent.outputCommands.extend(btagMuonInJetEventContent.outputCommands)
RECOSIMbtagMuonInJetEventContent.outputCommands.extend(RECOSIMEventContent.outputCommands)
RECOSIMbtagMuonInJetEventContent.outputCommands.extend(btagMuonInJetEventContent.outputCommands)

