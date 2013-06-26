import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from RecoBTag.Skimming.btagMC_QCD_20_50_EventContent_cff import *
btagMC_QCD_20_50OutputModule = cms.OutputModule("PoolOutputModule",
    btagMC_QCD_20_50EventSelection,
    FEVTSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('btagMC_QCD_20_50'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('btagMC_QCD_20_50.root')
)


