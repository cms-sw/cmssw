import FWCore.ParameterSet.Config as cms

from EgammaAnalysis.CSA07Skims.EgammaSkimEventContent_cff import *
from EgammaAnalysis.CSA07Skims.RECOSIMEgammaSkimEventContent_cff import *
egammaZPlusEMOrJetOutputModuleRECOSIM = cms.OutputModule("PoolOutputModule",
    RECOSIMEgammaSkimEventContent,
    egammaZPlusEMOrJetEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('egammaZPlusEMOrJetRECOSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('egammaZPlusEMOrJetRECOSIM.root')
)


