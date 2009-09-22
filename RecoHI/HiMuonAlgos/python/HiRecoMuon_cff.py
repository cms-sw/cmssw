import FWCore.ParameterSet.Config as cms

from RecoMuon.Configuration.RecoMuon_cff import *

hiTracks = 'hiGlobalPrimTracks' #heavy ion track label

# replace with heavy ion track label
muons.inputCollectionLabels = [hiTracks, 'globalMuons', 'standAloneMuons:UpdatedAtVtx']
muons.TrackExtractorPSet.inputTrackCollection = hiTracks
calomuons.inputTracks = hiTracks
muIsoDepositTk.ExtractorPSet.inputTrackCollection = hiTracks
globalMuons.TrackerCollectionLabel = hiTracks

# replace with heavy ion jet label
muons.JetExtractorPSet.JetCollectionLabel = cms.InputTag("iterativeConePu5CaloJets")
muIsoDepositJets.ExtractorPSet.JetCollectionLabel = cms.InputTag("iterativeConePu5CaloJets")

# HI muon sequences
muonRecoPbPb = cms.Sequence(muontracking_with_TeVRefinement
                            #* muonIdProducerSequence       #needs fixing
                            )

#muonRecoPbPbWithIsolation = cms.Sequence(muonRecoPbPb * muIsolation)  #needs testing
