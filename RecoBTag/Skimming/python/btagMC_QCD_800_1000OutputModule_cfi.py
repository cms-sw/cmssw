import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from RecoBTag.Skimming.btagMC_QCD_800_1000_EventContent_cff import *
btagMC_QCD_800_1000OutputModule = cms.OutputModule("PoolOutputModule",
    btagMC_QCD_800_1000EventSelection,
    FEVTSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('btagMC_QCD_800_1000'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('btagMC_QCD_800_1000.root')
)


# foo bar baz
