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
eras.trackingPhase1.toModify(
    earlyGeneralTracks,
    trackProducers = [
        'initialStepTracks',
        'highPtTripletStepTracks',
        'jetCoreRegionalStepTracks',
        'lowPtQuadStepTracks',
        'lowPtTripletStepTracks',
        'detachedQuadStepTracks',
        #'detachedTripletStepTracks', # FIXME: disabled for now
        'mixedTripletStepTracks',
        'pixelLessStepTracks',
        'tobTecStepTracks'
    ],
    inputClassifiers = [
        "initialStep",
        "highPtTripletStep",
        "jetCoreRegionalStep",
        "lowPtQuadStep",
        "lowPtTripletStep",
        "detachedQuadStep",
        #"detachedTripletStep", # FIXME: disabled for now
        "mixedTripletStep",
        "pixelLessStep",
        "tobTecStep"
    ],
)

# For Phase1PU70
from RecoTracker.FinalTrackSelectors.trackListMerger_cfi import trackListMerger as _trackListMerger
eras.trackingPhase1PU70.toReplaceWith(earlyGeneralTracks, _trackListMerger.clone(
    TrackProducers = ['initialStepTracks',
                      'highPtTripletStepTracks',
                      'lowPtQuadStepTracks',
                      'lowPtTripletStepTracks',
                      'detachedQuadStepTracks',
                      'mixedTripletStepTracks',
                      'pixelPairStepTracks',
                      'tobTecStepTracks'],
    hasSelector = [1,1,1,1,1,1,1,1],
    indivShareFrac = [1.0,0.16,0.095,0.09,0.095,0.095,0.095,0.08],
    selectedTrackQuals = [cms.InputTag("initialStepSelector","initialStep"),
                          cms.InputTag("highPtTripletStepSelector","highPtTripletStep"),
                          cms.InputTag("lowPtQuadStepSelector","lowPtQuadStep"),
                          cms.InputTag("lowPtTripletStepSelector","lowPtTripletStep"),
                          cms.InputTag("detachedQuadStep"),
                          cms.InputTag("mixedTripletStep"),
                          cms.InputTag("pixelPairStepSelector","pixelPairStep"),
                          cms.InputTag("tobTecStepSelector","tobTecStep")],
    setsToMerge = [cms.PSet( tLists=cms.vint32(0,1,2,3,4,5,6,7), pQual=cms.bool(True) ) ],
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
))
