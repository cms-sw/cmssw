import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToInvisible_EventContent_cff import *
higgsToInvisibleOutputModuleAODSIM = cms.OutputModule("PoolOutputModule",
    higgsToInvisibleEventSelection,
    AODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('higgsToInvisibleAODSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('hToInvis_AODSIM.root')
)


