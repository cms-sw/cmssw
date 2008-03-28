import FWCore.ParameterSet.Config as cms

from JetMETAnalysis.JetSkims.dijetbalance_EventContent_cff import *
from JetMETAnalysis.JetSkims.FEVTSIMDiJetBalanceEventContent_cff import *
dijetbalanceOutputModuleFEVTSIM = cms.OutputModule("PoolOutputModule",
    dijetbalanceEventSelection,
    FEVTSIMDiJetBalanceEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('dijetbalance_FEVTSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('dijetbalance_FEVTSIM.root')
)


