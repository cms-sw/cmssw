import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
AODSIMWJetEventContent = cms.OutputModule("PoolOutputModule",
    AODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('EgammaWJetFilter'),
        dataTier = cms.untracked.string('AODSIM')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('electronFilterPath', 
            'muonFilterPath')
    ),
    fileName = cms.untracked.string('WJetFilteredAODSIM.root')
)

RECOSIMWJetEventContent = cms.OutputModule("PoolOutputModule",
    RECOSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('EgammaWJetFilter'),
        dataTier = cms.untracked.string('RECOSIM')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('electronFilterWPath', 
            'muonFilterWPath')
    ),
    fileName = cms.untracked.string('WJetFilteredRECOSIM.root')
)


