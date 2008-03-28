import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from JetMETAnalysis.JetSkims.dijetbalance_EventContent_cff import *
FEVTSIMDiJetBalanceEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
FEVTSIMDiJetBalanceEventContent.outputCommands.extend(FEVTSIMEventContent.outputCommands)
FEVTSIMDiJetBalanceEventContent.outputCommands.extend(dijetbalanceEventContent.outputCommands)

