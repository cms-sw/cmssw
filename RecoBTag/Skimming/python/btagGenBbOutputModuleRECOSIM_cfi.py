import FWCore.ParameterSet.Config as cms

from RecoBTag.Skimming.btagGenBb_EventContent_cff import *
btagGenBbOutputModuleRECOSIM = cms.OutputModule("PoolOutputModule",
    RECOSIMbtagGenBbEventContent,
    btagGenBbEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('btagGenBbRECOSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('btagGenBbRECOSIM.root')
)


