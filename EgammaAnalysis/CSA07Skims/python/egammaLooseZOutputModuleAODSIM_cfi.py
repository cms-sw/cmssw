import FWCore.ParameterSet.Config as cms

from EgammaAnalysis.CSA07Skims.EgammaSkimEventContent_cff import *
from EgammaAnalysis.CSA07Skims.AODSIMEgammaSkimEventContent_cff import *
egammaLooseZOutputModuleAODSIM = cms.OutputModule("PoolOutputModule",
    AODSIMEgammaSkimEventContent,
    egammaLooseZEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('egammaLooseZAODSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('egammaLooseZAODSIM.root')
)


