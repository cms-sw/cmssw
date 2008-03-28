import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from RecoBTag.Skimming.btagMC_QCD_50_80_EventContent_cff import *
btagMC_QCD_50-80OutputModule = cms.OutputModule("PoolOutputModule",
    btagMC_QCD_50-80EventSelection,
    FEVTSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('btagMC_QCD_50-80'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('btagMC_QCD_50-80.root')
)


