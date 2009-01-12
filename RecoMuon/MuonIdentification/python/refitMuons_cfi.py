import FWCore.ParameterSet.Config as cms

# Produces a collection recoMuons_refitMuons__* consisting of the
# reco::Muons from the src collection (default is the merged muons
# collection, only taking those Muons for which isGlobalMuon() is
# true) with the kinematics replaced by those from the refit tracks
# chosen. The default below chooses the tracks using the "cocktail"
# function found in DataFormats/MuonReco/src/MuonCocktails.cc.

refitMuons = cms.EDProducer('MuonsFromRefitTracksProducer',
    src           = cms.InputTag('muons'),
    tevMuonTracks = cms.untracked.string('tevMuons'),
    fromCocktail  = cms.untracked.bool(True),
    fromTMR       = cms.untracked.bool(False),
    TMRcut        = cms.untracked.double(3.5)
)
