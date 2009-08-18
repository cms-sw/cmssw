import FWCore.ParameterSet.Config as cms

from RecoMuon.Configuration.RecoMuon_cff import *
muons.inputCollectionLabels = ['globalPrimTracks', 'globalMuons', 'standAloneMuons:UpdatedAtVtx']
muons.TrackExtractorPSet.inputTrackCollection = 'globalPrimTracks'
calomuons.inputTracks = 'globalPrimTracks'
muIsoDepositTk.ExtractorPSet.inputTrackCollection = 'globalPrimTracks'
globalMuons.TrackerCollectionLabel = 'globalPrimTracks'
