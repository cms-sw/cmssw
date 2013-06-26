import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
AODSIMZJetEventContent = cms.OutputModule("PoolOutputModule",
    AODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('EgammaZJetFilter'),
        dataTier = cms.untracked.string('AODSIM')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('electronFilterPath', 
            'muonFilterPath')
    ),
    fileName = cms.untracked.string('ZJetFilteredAODSIM.root')
)

RECOSIMZJetEventContent = cms.OutputModule("PoolOutputModule",
    RECOSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('EgammaZJetFilter'),
        dataTier = cms.untracked.string('RECOSIM')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('electronFilterZPath', 
            'muonFilterZPath')
    ),
    fileName = cms.untracked.string('ZJetFilteredRECOSIM.root')
)


