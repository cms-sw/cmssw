import FWCore.ParameterSet.Config as cms

from EgammaAnalysis.CSA07Skims.EgammaSkimEventContent_cff import *
from EgammaAnalysis.CSA07Skims.AODSIMEgammaSkimEventContent_cff import *
egammaZPlusEMOrJetOutputModuleAODSIM = cms.OutputModule("PoolOutputModule",
    AODSIMEgammaSkimEventContent,
    egammaZPlusEMOrJetEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('egammaZPlusEMOrJetAODSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('egammaZPlusEMOrJetAODSIM.root')
)


