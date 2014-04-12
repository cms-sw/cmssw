import FWCore.ParameterSet.Config as cms

from RecoBTag.Skimming.btagElecInJet_EventContent_cff import *
btagElecInJetOutputModuleRECOSIM = cms.OutputModule("PoolOutputModule",
    btagElecInJetEventSelection,
    RECOSIMbtagElecInJetEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('btagElecInJetRECOSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('btagElecInJetRECOSIM.root')
)


