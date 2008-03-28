import FWCore.ParameterSet.Config as cms

from EgammaAnalysis.CSA07Skims.EgammaSkimEventContent_cff import *
from EgammaAnalysis.CSA07Skims.RECOSIMEgammaSkimEventContent_cff import *
egammaVeryHighEtOutputModuleRECOSIM = cms.OutputModule("PoolOutputModule",
    RECOSIMEgammaSkimEventContent,
    egammaVeryHighEtEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('egammaVeryHighEtRECOSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('egammaVeryHighEtRECOSIM.root')
)


