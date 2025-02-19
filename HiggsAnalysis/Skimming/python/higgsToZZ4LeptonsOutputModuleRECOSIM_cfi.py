import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToZZ4Leptons_EventContent_cff import *
higgsToZZ4LeptonsOutputModuleRECOSIM = cms.OutputModule("PoolOutputModule",
    RECOSIMEventContent,
    higgsToZZ4LeptonsEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('higgsToZZ4LeptonsRECOSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('hzz4l_RECOSIM.root')
)


