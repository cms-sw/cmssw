import FWCore.ParameterSet.Config as cms

#include "Configuration/EventContent/data/EventContent.cff"
from HeavyFlavorAnalysis.Skimming.onia_EventContent_cff import *
jpsiToMuMuOutputModuleAODSIM = cms.OutputModule("PoolOutputModule",
    AODSIMEventContent,
    jpsiToMuMuEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('jpsiToMuMu'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('jpsiToMuMu.root')
)


