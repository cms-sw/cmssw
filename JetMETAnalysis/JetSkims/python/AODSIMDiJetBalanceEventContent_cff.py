import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from JetMETAnalysis.JetSkims.dijetbalance_EventContent_cff import *
AODSIMDiJetBalanceEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
AODSIMDiJetBalanceEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
AODSIMDiJetBalanceEventContent.outputCommands.extend(dijetbalanceEventContent.outputCommands)

