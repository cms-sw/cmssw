import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from JetMETAnalysis.JetSkims.onejet_EventContent_cff import *
AODSIMOneJetEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
AODSIMOneJetEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
AODSIMOneJetEventContent.outputCommands.extend(onejetEventContent.outputCommands)

