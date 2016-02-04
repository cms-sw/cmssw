import FWCore.ParameterSet.Config as cms

from EgammaAnalysis.CSA07Skims.EgammaSkimEventContent_cff import *
from EgammaAnalysis.CSA07Skims.AODSIMEgammaSkimEventContent_cff import *
egammaVeryHighEtOutputModuleAODSIM = cms.OutputModule("PoolOutputModule",
    AODSIMEgammaSkimEventContent,
    egammaVeryHighEtEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('egammaVeryHighEtAODSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('egammaVeryHighEtAODSIM.root')
)


