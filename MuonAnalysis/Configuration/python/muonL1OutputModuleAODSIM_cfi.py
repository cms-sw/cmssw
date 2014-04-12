import FWCore.ParameterSet.Config as cms

from MuonAnalysis.Configuration.muonL1_EventContent_cff import *
muonL1OutputModuleAODSIM = cms.OutputModule("PoolOutputModule",
    muonL1EventSelection,
    AODSIMmuonL1EventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('muonL1AODSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('muonL1-AODSIM.root')
)


