import FWCore.ParameterSet.Config as cms

from EgammaAnalysis.CSA07Skims.EgammaSkimEventContent_cff import *
from EgammaAnalysis.CSA07Skims.RECOSIMEgammaSkimEventContent_cff import *
egammaWPlusEMOrJetOutputModuleRECOSIM = cms.OutputModule("PoolOutputModule",
    egammaWPlusEMOrJetEventSelection,
    RECOSIMEgammaSkimEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('egammaWPlusEMOrJetRECOSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('egammaWPlusEMOrJetRECOSIM.root')
)


