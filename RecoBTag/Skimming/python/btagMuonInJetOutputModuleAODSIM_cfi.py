import FWCore.ParameterSet.Config as cms

from RecoBTag.Skimming.btagMuonInJet_EventContent_cff import *
btagMuonInJetOutputModuleAODSIM = cms.OutputModule("PoolOutputModule",
    btagMuonInJetEventSelection,
    AODSIMbtagMuonInJetEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('btagMuonInJetAODSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('btagMuonInJetAODSIM.root')
)


