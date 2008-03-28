import FWCore.ParameterSet.Config as cms

from JetMETAnalysis.JetSkims.dijetbalance_EventContent_cff import *
from JetMETAnalysis.JetSkims.AODSIMDiJetBalanceEventContent_cff import *
from JetMETAnalysis.JetSkims.RECOSIMDiJetBalanceEventContent_cff import *
from JetMETAnalysis.JetSkims.FEVTSIMDiJetBalanceEventContent_cff import *
dijetbalanceOutputModuleAODSIM = cms.OutputModule("PoolOutputModule",
    dijetbalanceEventSelection,
    AODSIMDiJetBalanceEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('dijetbalance_AODSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('dijetbalance_AODSIM.root')
)

dijetbalanceOutputModuleRECOSIM = cms.OutputModule("PoolOutputModule",
    dijetbalanceEventSelection,
    RECOSIMDiJetBalanceEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('dijetbalance_RECOSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('dijetbalance_RECOSIM.root')
)

dijetbalanceOutputModuleFEVTSIM = cms.OutputModule("PoolOutputModule",
    dijetbalanceEventSelection,
    FEVTSIMDiJetBalanceEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('dijetbalance_FEVTSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('dijetbalance_FEVTSIM.root')
)


