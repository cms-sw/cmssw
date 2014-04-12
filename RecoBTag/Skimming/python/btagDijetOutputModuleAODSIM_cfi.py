import FWCore.ParameterSet.Config as cms

from RecoBTag.Skimming.btagDijet_EventContent_cff import *
btagDijetOutputModuleAODSIM = cms.OutputModule("PoolOutputModule",
    AODSIMbtagDijetEventContent,
    btagDijetEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('btagDijetAODSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('btagDijetAODSIM.root')
)


