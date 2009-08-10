import FWCore.ParameterSet.Config as cms

from HiggsAnalysis.Skimming.higgsToTauTau_MuonTau_EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToTauTau_MuonTau_EventContent_AODSIM_cff import *
from HiggsAnalysis.Skimming.higgsToTauTau_MuonTau_EventContent_RECOSIM_cff import *
higgsToTauTauMuonTauOutputModuleRECOSIM = cms.OutputModule("PoolOutputModule",
    higgsToTauTauMuonTauEventSelection,
    higgsToTauTauMuonTauEventContentRECOSIM,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('higgsToTauTauMuonTauRECOSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('higgsToTauTauMuonTauRECOSIM.root')
)


