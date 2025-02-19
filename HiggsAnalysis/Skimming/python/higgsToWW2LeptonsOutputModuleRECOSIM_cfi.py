import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToWW2Leptons_EventContent_cff import *
higgsToWW2LeptonsOutputModuleRECOSIM = cms.OutputModule("PoolOutputModule",
    RECOSIMEventContent,
    higgsToWW2LeptonsEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('higgsToWW2LeptonsRECOSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('hww2l_RECOSIM.root')
)


