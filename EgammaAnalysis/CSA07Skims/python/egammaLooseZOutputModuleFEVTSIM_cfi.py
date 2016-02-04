import FWCore.ParameterSet.Config as cms

from EgammaAnalysis.CSA07Skims.EgammaSkimEventContent_cff import *
from EgammaAnalysis.CSA07Skims.FEVTSIMEgammaSkimEventContent_cff import *
egammaLooseZOutputModuleFEVTSIM = cms.OutputModule("PoolOutputModule",
    FEVTSIMEgammaSkimEventContent,
    egammaLooseZEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('egammaLooseZFEVTSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('egammaLooseZFEVTSIM.root')
)


