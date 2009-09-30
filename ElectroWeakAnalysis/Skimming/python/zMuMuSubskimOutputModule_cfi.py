import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *

zMuMuSubskimOutputModule = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
      'drop *',
      'keep *_selectedLayer1MuonsTriggerMatch_*_*',
      'keep *_selectedLayer1TrackCands_*_*',
      'keep *_dimuons_*_*',
      'keep *_dimuonsOneTrack_*_*',
      'keep *_dimuonsGlobal_*_*',
      'keep *_dimuonsOneStandAloneMuon_*_*',
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(
           'dimuonsPath',
           'dimuonsOneTrackPath')
    ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('zmumu'),
        dataTier = cms.untracked.string('USER')
   ),
   fileName = cms.untracked.string('zMuMuSubskim.root')
)

