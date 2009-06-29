import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.lightChHiggsToTauNu_EventContent_cff import *
lightChHiggsToTauNuOutputModuleAODSIM = cms.OutputModule("PoolOutputModule",
    lightChHiggsToTauNuEventSelection,
    AODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('lightChHiggsToTauNuAODSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('lightChHiggsToTauNu_AODSIM.root')
)



