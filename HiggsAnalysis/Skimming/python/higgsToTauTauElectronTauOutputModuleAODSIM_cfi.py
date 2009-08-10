import FWCore.ParameterSet.Config as cms

from HiggsAnalysis.Skimming.higgsToTauTau_ElectronTau_EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToTauTau_ElectronTau_EventContent_AODSIM_cff import *
higgsToTauTauElectronTauOutputModuleAODSIM = cms.OutputModule("PoolOutputModule",
    higgsToTauTauElectronTauEventSelection,
    higgsToTauTauElectronTauEventContentAODSIM,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('higgsToTauTauElectronTauAODSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('higgsToTauTauElectronTauAODSIM.root')
)


