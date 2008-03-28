import FWCore.ParameterSet.Config as cms

from EgammaAnalysis.CSA07Skims.EgammaSkimEventContent_cff import *
#
# EgammaAnalysis event content
# 
# the accumulated event contents should not be used, 
# but it's important that all EventContent definitions are included here
#
EgammaAnalysisEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
EgammaAnalysisEventContent.outputCommands.extend(egammaSkimEventContent.outputCommands)

