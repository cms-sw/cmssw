import FWCore.ParameterSet.Config as cms

from JetMETAnalysis.JetSkims.onejet_EventContent_cff import *
from JetMETAnalysis.JetSkims.FEVTSIMOneJetEventContent_cff import *
onejetOutputModuleFEVTSIM = cms.OutputModule("PoolOutputModule",
    FEVTSIMOneJetEventContent,
    onejetEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('onejet_FEVTSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('onejet_FEVTSIM.root')
)


