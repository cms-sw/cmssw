import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *

zMuMuSubskimOutputModule = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
      'drop *',
####   to access the position at the momentum system for global and standalone muons
      'keep recoTrackExtras_standAloneMuons_*_*',
      'keep recoTracks_standAloneMuons_UpdatedAtVtx_*',
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

