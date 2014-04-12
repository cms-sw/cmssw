import FWCore.ParameterSet.Config as cms

from HiggsAnalysis.Skimming.higgsToTauTau_LeptonTau_EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToTauTau_LeptonTau_EventContent_AODSIM_cff import *
higgsToTauTauLeptonTauOutputModuleAODSIM = cms.OutputModule("PoolOutputModule",
    higgsToTauTauLeptonTauEventSelection,
    higgsToTauTauLeptonTauEventContentAODSIM,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('higgsToTauTauLeptonTauAODSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('higgsToTauTauLeptonTauAODSIM.root')
)


