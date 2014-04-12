import FWCore.ParameterSet.Config as cms

from RecoBTag.Skimming.btagGenBb_EventContent_cff import *
btagGenBbOutputModuleAODSIM = cms.OutputModule("PoolOutputModule",
    AODSIMbtagGenBbEventContent,
    btagGenBbEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('btagGenBbAODSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('btagGenBbAODSIM.root')
)


