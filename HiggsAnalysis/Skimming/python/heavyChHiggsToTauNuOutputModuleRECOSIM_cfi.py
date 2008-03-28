import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.heavyChHiggsToTauNu_EventContent_cff import *
heavyChHiggsToTauNuOutputModuleRECOSIM = cms.OutputModule("PoolOutputModule",
    RECOSIMEventContent,
    heavyChHiggsToTauNuEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('heavyChHiggsToTauNuRECOSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('heavyChHiggsToTauNu_RECOSIM.root')
)


