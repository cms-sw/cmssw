import FWCore.ParameterSet.Config as cms

from RecoBTag.Skimming.btagMuonInJet_EventContent_cff import *
btagMuonInJetOutputModuleRECOSIM = cms.OutputModule("PoolOutputModule",
    btagMuonInJetEventSelection,
    RECOSIMbtagMuonInJetEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('btagMuonInJetRECOSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('btagMuonInJetRECOSIM.root')
)


