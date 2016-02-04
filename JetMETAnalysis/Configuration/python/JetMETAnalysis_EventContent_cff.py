import FWCore.ParameterSet.Config as cms

# MET Event Content
from JetMETAnalysis.METSkims.metHigh_EventContent_cff import *
from JetMETAnalysis.METSkims.metLow_EventContent_cff import *

# JET Event Content
#from JetMETAnalysis.JetSkims.onejet_EventContent_cff import *
#from JetMETAnalysis.JetSkims.photonjets_EventContent_cff import *

JetMETAnalysisEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
JetMETAnalysisEventContent.outputCommands.extend(metHighEventContent.outputCommands)
JetMETAnalysisEventContent.outputCommands.extend(metLowEventContent.outputCommands)
#JetMETAnalysisEventContent.outputCommands.extend(onejetEventContent.outputCommands)
#JetMETAnalysisEventContent.outputCommands.extend(photonjetsEventContent.outputCommands)

