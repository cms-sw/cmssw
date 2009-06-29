import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.lightChHiggsToTauNu_EventContent_cff import *
lightChHiggsToTauNuOutputModuleRECOSIM = cms.OutputModule("PoolOutputModule",
    RECOSIMEventContent,
    lightChHiggsToTauNuEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('lightChHiggsToTauNuRECOSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('lightChHiggsToTauNu_RECOSIM.root')
)



