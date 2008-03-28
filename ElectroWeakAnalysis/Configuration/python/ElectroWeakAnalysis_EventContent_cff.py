import FWCore.ParameterSet.Config as cms

#
# ElectroWeakAnalysis event content 
#
from ElectroWeakAnalysis.ZReco.zToMuMu_EventContent_cff import *
from ElectroWeakAnalysis.ZReco.zToMuMuGolden_EventContent_cff import *
from ElectroWeakAnalysis.ZReco.zToEE_EventContent_cff import *
from ElectroWeakAnalysis.ZReco.zToTauTau_ETau_EventContent_cff import *
ElectroWeakAnalysisEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
ElectroWeakAnalysisEventContent.outputCommands.extend(zToMuMuEventContent.outputCommands)
ElectroWeakAnalysisEventContent.outputCommands.extend(zToMuMuGoldenEventContent.outputCommands)
ElectroWeakAnalysisEventContent.outputCommands.extend(zToEEEventContent.outputCommands)
ElectroWeakAnalysisEventContent.outputCommands.extend(zToTauTauETauEventContent.outputCommands)

