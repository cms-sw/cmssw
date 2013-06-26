import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToInvisible_EventContent_cff import *
higgsToInvisibleOutputModuleRECOSIM = cms.OutputModule("PoolOutputModule",
    RECOSIMEventContent,
    higgsToInvisibleEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('higgsToInvisibleRECOSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('hToInvis_RECOSIM.root')
)


