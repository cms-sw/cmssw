import FWCore.ParameterSet.Config as cms
from RecoTracker.FinalTrackSelectors.TrackCollectionMerger_cfi import *
from RecoTracker.FinalTrackSelectors.trackAlgoPriorityOrder_cfi import trackAlgoPriorityOrder

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
displacedRegionalStepInputTracks =  TrackCollectionMerger.clone(
    trackProducers = [
        'earlyGeneralTracks',
        'muonSeededTracksInOut',
        'muonSeededTracksOutIn'
    ],
    inputClassifiers =[
        "earlyGeneralTracks",
        "muonSeededTracksInOutClassifier",
        "muonSeededTracksOutInClassifier"
    ]
)
