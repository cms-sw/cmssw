import FWCore.ParameterSet.Config as cms

from RecoMuon.Configuration.RecoMuon_cff import *

hiTracks = 'hiGlobalPrimTracks' #heavy ion track label

# replace with heavy ion track label
muons.inputCollectionLabels = [hiTracks, 'globalMuons', 'standAloneMuons:UpdatedAtVtx']
muons.TrackExtractorPSet.inputTrackCollection = hiTracks

calomuons.inputTracks = hiTracks

globalMuons.TrackerCollectionLabel = hiTracks

# replace with heavy ion jet label
muons.JetExtractorPSet.JetCollectionLabel = cms.InputTag("iterativeConePu5CaloJets")

# turn off calo muons for timing considerations?
#muons.fillCaloCompatibility = cms.bool(False)

# HI muon sequence (passed to RecoHI.Configuration.Reconstruction_HI_cff)
muonRecoPbPb = cms.Sequence(muonreco_plus_isolation)

