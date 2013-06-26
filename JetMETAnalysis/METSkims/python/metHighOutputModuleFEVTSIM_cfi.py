import FWCore.ParameterSet.Config as cms

from JetMETAnalysis.METSkims.metHigh_EventContent_cff import *
from JetMETAnalysis.METSkims.FEVTSIMMetHigh_EventContent_cff import *
metHighOutputModuleFEVTSIM = cms.OutputModule("PoolOutputModule",
    FEVTSIMMetHighEventContent,
    metHighEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('metHigh_FEVTSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('metHigh_FEVTSIM.root')
)


