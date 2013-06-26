import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from RecoBTag.Skimming.btagMC_ttbar_EventContent_cff import *
btagMC_ttbarOutputModule = cms.OutputModule("PoolOutputModule",
    btagMC_ttbarEventSelection,
    FEVTSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('btagMC_ttbar'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('btagMC_ttbar.root')
)


