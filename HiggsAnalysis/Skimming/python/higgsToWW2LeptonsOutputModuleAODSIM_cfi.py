import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToWW2Leptons_EventContent_cff import *
higgsToWW2LeptonsOutputModuleAODSIM = cms.OutputModule("PoolOutputModule",
    AODSIMEventContent,
    higgsToWW2LeptonsEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('higgsToWW2LeptonsAODSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('hww2l_AODSIM.root')
)


