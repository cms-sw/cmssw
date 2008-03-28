import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from RecoBTag.Skimming.btagMC_QCD_120_170_EventContent_cff import *
btagMC_QCD_120-170OutputModule = cms.OutputModule("PoolOutputModule",
    btagMC_QCD_120-170EventSelection,
    FEVTSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('btagMC_QCD_120-170'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('btagMC_QCD_120-170.root')
)


