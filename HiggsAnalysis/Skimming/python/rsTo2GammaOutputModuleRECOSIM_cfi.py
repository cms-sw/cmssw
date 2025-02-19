import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.rsTo2Gamma_EventContent_cff import *
rsTo2GammaOutputModuleRECOSIM = cms.OutputModule("PoolOutputModule",
    RECOSIMEventContent,
    rsTo2GammaEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('rsTo2Gamma_RECOSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('rsgg_RECOSIM.root')
)


