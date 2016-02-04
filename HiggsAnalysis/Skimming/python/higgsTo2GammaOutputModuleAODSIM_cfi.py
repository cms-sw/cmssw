import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.higgsTo2Gamma_EventContent_cff import *
higgsTo2GammaOutputModuleAODSIM = cms.OutputModule("PoolOutputModule",
    higgsTo2GammaEventSelection,
    AODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('higgsTo2GammaAODSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('hgg_AODSIM.root')
)


