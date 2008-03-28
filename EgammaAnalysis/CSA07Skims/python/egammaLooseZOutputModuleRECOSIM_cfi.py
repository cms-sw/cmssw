import FWCore.ParameterSet.Config as cms

from EgammaAnalysis.CSA07Skims.EgammaSkimEventContent_cff import *
from EgammaAnalysis.CSA07Skims.RECOSIMEgammaSkimEventContent_cff import *
egammaLooseZOutputModuleRECOSIM = cms.OutputModule("PoolOutputModule",
    RECOSIMEgammaSkimEventContent,
    egammaLooseZEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('egammaLooseZRECOSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('egammaLooseZRECOSIM.root')
)


