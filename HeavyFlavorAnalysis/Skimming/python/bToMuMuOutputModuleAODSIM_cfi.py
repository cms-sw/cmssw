import FWCore.ParameterSet.Config as cms

#include "Configuration/EventContent/data/EventContent.cff"
from HeavyFlavorAnalysis.Skimming.onia_EventContent_cff import *
bToMuMuOutputModuleAODSIM = cms.OutputModule("PoolOutputModule",
    AODSIMEventContent,
    bToMuMuEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('bToMuMu'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('bToMuMu.root')
)


