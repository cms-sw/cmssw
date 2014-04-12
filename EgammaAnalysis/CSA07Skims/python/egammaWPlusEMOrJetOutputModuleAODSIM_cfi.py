import FWCore.ParameterSet.Config as cms

from EgammaAnalysis.CSA07Skims.EgammaSkimEventContent_cff import *
from EgammaAnalysis.CSA07Skims.AODSIMEgammaSkimEventContent_cff import *
egammaWPlusEMOrJetOutputModuleAODSIM = cms.OutputModule("PoolOutputModule",
    egammaWPlusEMOrJetEventSelection,
    AODSIMEgammaSkimEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('egammaWPlusEMOrJetAODSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('egammaWPlusEMOrJetAODSIM.root')
)


