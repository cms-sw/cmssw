import FWCore.ParameterSet.Config as cms

from RecoTracker.FinalTrackSelectors.singleLongTrackProducer_cfi import singleLongTrackProducer 

SingleLongTrackProducer = singleLongTrackProducer.clone(
    allTracks = "generalTracks",
    matchMuons = "earlyMuons",
    requiredDr= 0.01,
    minNumberOfLayers = 10,
    onlyValidHits = True,
    debug = False,
    minPt = 15.0,
    maxEta = 2.2,
    maxDxy = 0.02,
    maxDz = 0.5,
    PrimaryVertex = "offlinePrimaryVertices")
