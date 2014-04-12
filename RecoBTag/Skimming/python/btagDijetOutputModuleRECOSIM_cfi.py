import FWCore.ParameterSet.Config as cms

from RecoBTag.Skimming.btagDijet_EventContent_cff import *
btagDijetOutputModuleRECOSIM = cms.OutputModule("PoolOutputModule",
    RECOSIMbtagDijetEventContent,
    btagDijetEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('btagDijetRECOSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('btagDijetRECOSIM.root')
)


