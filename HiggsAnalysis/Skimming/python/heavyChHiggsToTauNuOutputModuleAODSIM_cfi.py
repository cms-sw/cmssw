import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.heavyChHiggsToTauNu_EventContent_cff import *
heavyChHiggsToTauNuOutputModuleAODSIM = cms.OutputModule("PoolOutputModule",
    AODSIMEventContent,
    heavyChHiggsToTauNuEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('heavyChHiggsToTauNuAODSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('heavyChHiggsToTauNu_AODSIM.root')
)


