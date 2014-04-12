import FWCore.ParameterSet.Config as cms

from RecoBTag.Skimming.btagElecInJet_EventContent_cff import *
btagElecInJetOutputModuleAODSIM = cms.OutputModule("PoolOutputModule",
    btagElecInJetEventSelection,
    AODSIMbtagElecInJetEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('btagElecInJetAODSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('btagElecInJetAODSIM.root')
)


