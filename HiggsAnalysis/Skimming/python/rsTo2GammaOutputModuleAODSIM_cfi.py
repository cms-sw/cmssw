import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.rsTo2Gamma_EventContent_cff import *
rsTo2GammaOutputModuleAODSIM = cms.OutputModule("PoolOutputModule",
    rsTo2GammaEventSelection,
    AODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('rsTo2GammaAODSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('rsgg_AODSIM.root')
)


