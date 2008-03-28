import FWCore.ParameterSet.Config as cms

from JetMETAnalysis.JetSkims.dijetbalance_EventContent_cff import *
from JetMETAnalysis.JetSkims.RECOSIMDiJetBalanceEventContent_cff import *
dijetbalanceOutputModuleRECOSIM = cms.OutputModule("PoolOutputModule",
    dijetbalanceEventSelection,
    RECOSIMDiJetBalanceEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('dijetbalance_RECOSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('dijetbalance_RECOSIM.root')
)


