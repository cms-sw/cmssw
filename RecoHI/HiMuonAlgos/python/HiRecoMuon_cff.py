import FWCore.ParameterSet.Config as cms

from RecoMuon.Configuration.RecoMuon_cff import *
muons.inputCollectionLabels = ['hiGlobalPrimTracks', 'globalMuons', 'standAloneMuons:UpdatedAtVtx']
muons.TrackExtractorPSet.inputTrackCollection = 'hiGlobalPrimTracks'
calomuons.inputTracks = 'hiGlobalPrimTracks'
muIsoDepositTk.ExtractorPSet.inputTrackCollection = 'hiGlobalPrimTracks'
globalMuons.TrackerCollectionLabel = 'hiGlobalPrimTracks'
