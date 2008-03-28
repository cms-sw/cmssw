import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToZZ4Leptons_EventContent_cff import *
higgsToZZ4LeptonsOutputModuleAODSIM = cms.OutputModule("PoolOutputModule",
    higgsToZZ4LeptonsEventSelection,
    AODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('higgsToZZ4LeptonsAODSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('hzz4l_AODSIM.root')
)


