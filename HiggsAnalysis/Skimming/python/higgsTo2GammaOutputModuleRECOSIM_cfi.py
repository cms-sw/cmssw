import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.higgsTo2Gamma_EventContent_cff import *
higgsTo2GammaOutputModuleRECOSIM = cms.OutputModule("PoolOutputModule",
    RECOSIMEventContent,
    higgsTo2GammaEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('higgsTo2Gamma_RECOSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('hgg_RECOSIM.root')
)


