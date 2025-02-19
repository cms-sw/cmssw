import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from RecoBTag.Skimming.btagMC_QCD_80_120_EventContent_cff import *
btagMC_QCD_80_120OutputModule = cms.OutputModule("PoolOutputModule",
    btagMC_QCD_80_120EventSelection,
    FEVTSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('btagMC_QCD_80_120'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('btagMC_QCD_80_120.root')
)


