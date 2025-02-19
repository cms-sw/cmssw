import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from RecoBTag.Skimming.btagMC_QCD_200_300_EventContent_cff import *
btagMC_QCD_200_300OutputModule = cms.OutputModule("PoolOutputModule",
    btagMC_QCD_200_300EventSelection,
    FEVTSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('btagMC_QCD_200_300'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('btagMC_QCD_200_300.root')
)


