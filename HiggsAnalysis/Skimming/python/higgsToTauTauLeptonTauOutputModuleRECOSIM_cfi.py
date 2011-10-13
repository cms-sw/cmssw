import FWCore.ParameterSet.Config as cms

from HiggsAnalysis.Skimming.higgsToTauTau_LeptonTau_EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToTauTau_LeptonTau_EventContent_AODSIM_cff import *
from HiggsAnalysis.Skimming.higgsToTauTau_LeptonTau_EventContent_RECOSIM_cff import *
higgsToTauTauLeptonTauOutputModuleRECOSIM = cms.OutputModule("PoolOutputModule",
    higgsToTauTauLeptonTauEventSelection,
    higgsToTauTauLeptonTauEventContentRECOSIM,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('higgsToTauTauLeptonTauRECOSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('higgsToTauTauLeptonTauRECOSIM.root')
)


