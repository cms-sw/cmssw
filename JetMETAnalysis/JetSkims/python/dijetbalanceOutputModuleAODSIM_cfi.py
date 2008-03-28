import FWCore.ParameterSet.Config as cms

from JetMETAnalysis.JetSkims.dijetbalance_EventContent_cff import *
from JetMETAnalysis.JetSkims.AODSIMDiJetBalanceEventContent_cff import *
dijetbalanceOutputModuleAODSIM = cms.OutputModule("PoolOutputModule",
    dijetbalanceEventSelection,
    AODSIMDiJetBalanceEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('dijetbalance_AODSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('dijetbalance_AODSIM.root')
)


