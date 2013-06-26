import FWCore.ParameterSet.Config as cms

from MuonAnalysis.Configuration.muonL1_EventContent_cff import *
muonL1OutputModuleRECOSIM = cms.OutputModule("PoolOutputModule",
    muonL1EventSelection,
    RECOSIMmuonL1EventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('muonL1RECOSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('muonL1-RECOSIM.root')
)


