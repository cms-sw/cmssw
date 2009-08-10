import FWCore.ParameterSet.Config as cms

from HiggsAnalysis.Skimming.higgsToTauTau_MuonTau_EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToTauTau_MuonTau_EventContent_AODSIM_cff import *
higgsToTauTauMuonTauOutputModuleAODSIM = cms.OutputModule("PoolOutputModule",
    higgsToTauTauMuonTauEventSelection,
    higgsToTauTauMuonTauEventContentAODSIM,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('higgsToTauTauMuonTauAODSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('higgsToTauTauMuonTauAODSIM.root')
)


