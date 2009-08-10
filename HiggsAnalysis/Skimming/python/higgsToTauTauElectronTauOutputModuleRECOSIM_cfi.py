import FWCore.ParameterSet.Config as cms

from HiggsAnalysis.Skimming.higgsToTauTau_ElectronTau_EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToTauTau_ElectronTau_EventContent_AODSIM_cff import *
from HiggsAnalysis.Skimming.higgsToTauTau_ElectronTau_EventContent_RECOSIM_cff import *
higgsToTauTauElectronTauOutputModuleRECOSIM = cms.OutputModule("PoolOutputModule",
    higgsToTauTauElectronTauEventSelection,
    higgsToTauTauElectronTauEventContentRECOSIM,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('higgsToTauTauElectronTauRECOSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('higgsToTauTauElectronTauRECOSIM.root')
)


