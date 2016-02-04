import FWCore.ParameterSet.Config as cms

#include "Configuration/EventContent/data/EventContent.cff"
from HeavyFlavorAnalysis.Skimming.onia_EventContent_cff import *
upsilonToMuMuOutputModuleAODSIM = cms.OutputModule("PoolOutputModule",
    upsilonToMuMuEventSelection,
    AODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('upsilonToMuMu'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('upsilonToMuMu.root')
)


