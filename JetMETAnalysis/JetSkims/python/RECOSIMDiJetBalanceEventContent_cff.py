import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from JetMETAnalysis.JetSkims.dijetbalance_EventContent_cff import *
RECOSIMDiJetBalanceEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RECOSIMDiJetBalanceEventContent.outputCommands.extend(RECOSIMEventContent.outputCommands)
RECOSIMDiJetBalanceEventContent.outputCommands.extend(dijetbalanceEventContent.outputCommands)

