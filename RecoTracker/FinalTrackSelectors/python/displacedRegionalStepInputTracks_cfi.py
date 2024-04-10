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
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
(pp_on_AA | pp_on_XeXe_2017).toModify(displacedRegionalStepInputTracks,
    trackProducers = [],
    inputClassifiers = []
)
