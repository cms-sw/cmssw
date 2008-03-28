import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
AODSIMWToMuNuEventContentTest = cms.OutputModule("PoolOutputModule",
    AODSIMEventContent,
    filterName = cms.untracked.string('WMuNuFilter'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('WMuNuFilterPath1muIso', 'WMuNuFilterPath1muNoIso')
    ),
    dataTier = cms.untracked.string('USER'),
    fileName = cms.untracked.string('/tmp/etorassa/WMuNuFiltered.root')
)


