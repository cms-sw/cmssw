import FWCore.ParameterSet.Config as cms

from HeavyFlavorAnalysis.Skimming.tauTo3Mu_EventContent_cff import *
from HeavyFlavorAnalysis.Skimming.AODSIMTauTo3Mu_EventContent_cff import *
tauTo3MuOutputModule = cms.OutputModule("PoolOutputModule",
    AODSIMTauTo3MuEventContent,
    tauTo3MuEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('tauTo3Mu'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('tauTo3Mu.root')
)


