import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
from RecoTracker.FinalTrackSelectors.TrackCollectionMerger_cfi import *

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
earlyGeneralTracks =  TrackCollectionMerger.clone()
earlyGeneralTracks.trackProducers = ['initialStepTracks',
                                     'jetCoreRegionalStepTracks',
                                     'lowPtTripletStepTracks',
                                     'pixelPairStepTracks',
                                     'detachedTripletStepTracks',
                                     'mixedTripletStepTracks',
                                     'pixelLessStepTracks',
                                     'tobTecStepTracks'
                                     ]
earlyGeneralTracks.inputClassifiers =["initialStep",
                                      "jetCoreRegionalStep",
                                      "lowPtTripletStep",
                                      "pixelPairStep",
                                      "detachedTripletStep",
                                      "mixedTripletStep",
                                      "pixelLessStep",
                                      "tobTecStep"
                                      ]

eras.trackingLowPU.toModify(earlyGeneralTracks,
    trackProducers = [
        'initialStepTracks',
        'lowPtTripletStepTracks',
        'pixelPairStepTracks',
        'detachedTripletStepTracks',
        'mixedTripletStepTracks',
        'pixelLessStepTracks',
        'tobTecStepTracks'
    ],
    inputClassifiers = [
        "initialStepSelector",
        "lowPtTripletStepSelector",
        "pixelPairStepSelector",
        "detachedTripletStep",
        "mixedTripletStep",
        "pixelLessStepSelector",
        "tobTecStep"
    ]
)
