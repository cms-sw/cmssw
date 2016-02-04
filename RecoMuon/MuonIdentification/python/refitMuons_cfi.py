import FWCore.ParameterSet.Config as cms

# Produces a collection recoMuons_refitMuons__* consisting of the
# reco::Muons from the src collection (default is the merged muons
# collection, only taking those Muons for which isGlobalMuon() is
# true) with the kinematics replaced by those from the refit tracks
# chosen. The default below chooses the tracks using the "cocktail"
# function found in DataFormats/MuonReco/src/MuonCocktails.cc.

refitMuons = cms.EDProducer('MuonsFromRefitTracksProducer',
    # The input MuonCollection from which the starting Muon objects
    # will be taken. The module will only consider globalMuons from
    # the merged muon collection, i.e. trackerMuons, stand-alone muons
    # will be filtered out of the merged MuonCollection.
    src = cms.InputTag('muons1stStep'),

    # The particular set of refit tracks to use. Could also be
    # 'tevMuons:default', 'tevMuons:picky', or 'tevMuons:firstHit' to
    # use the corresponding refits; 'none' to use this module as just
    # a filter for globalMuons (as opposed to trackerMuons or
    # caloMuons); to make Muons out of the cocktail tracks, 'tevMuons'
    # by itself must be used (also specifying fromCocktail = True
    # below).
    tevMuonTracks = cms.string('tevMuons'),

    # Exactly one of the below boolean flags may be True (determines
    # the refit track picked for each muon).

    # Whether to call muon::tevOptimized as in the above code and use
    # the result of the cocktail choice.
    fromCocktail = cms.bool(True),

    # Whether to replace the input muons' kinematics with that of the
    # tracker-only fit. I.e., the muon's momentum, vertex, and charge are
    # taken from the track accessed by reco::Muon::innerTrack().
    fromTrackerTrack = cms.bool(False),

    # Whether to replace the input muons' kinematics with that of the
    # tracker-only fit. I.e., the muon's momentum, vertex, and charge are
    # taken from the track accessed by reco::Muon::innerTrack().
    fromGlobalTrack = cms.bool(False),

    # Whether to apply the TMR cocktail algorithm. For each muon track, we
    # start with the first-muon-hit fit. If the difference in -ln(chi^2
    # tail probability) between the first-muon-hit and tracker-only is
    # greater than the below prescribed cut value, the tracker-only fit
    # replaces the first-muon-hit. For further details see XXX.
    fromTMR = cms.bool(False),

    # The cut value used in the TMR cocktail.
    TMRcut = cms.double(4.0),

    # Whether to use Adam Everett's sigma-switch method, choosing
    # between the global track and the tracker track.
    fromSigmaSwitch = cms.bool(False),

    # The number of sigma to switch on in the above method.
    nSigmaSwitch = cms.double(2),
    
    # The pT threshold to switch at in the above method.
    ptThreshold = cms.double(200),
)
