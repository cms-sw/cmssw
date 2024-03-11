import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from RecoBTag.Skimming.btagMC_QCD_120_170_EventContent_cff import *
btagMC_QCD_120_170OutputModule = cms.OutputModule("PoolOutputModule",
    btagMC_QCD_120_170EventSelection,
    FEVTSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('btagMC_QCD_120_170'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('btagMC_QCD_120_170.root')
)


# foo bar baz
