import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from JetMETAnalysis.JetSkims.onejet_EventContent_cff import *
RECOSIMOneJetEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RECOSIMOneJetEventContent.outputCommands.extend(RECOSIMEventContent.outputCommands)
RECOSIMOneJetEventContent.outputCommands.extend(onejetEventContent.outputCommands)

