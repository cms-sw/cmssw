import FWCore.ParameterSet.Config as cms
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
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toModify(earlyGeneralTracks,
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
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
from Configuration.Eras.Modifier_trackingPhase1QuadProp_cff import trackingPhase1QuadProp
_forPhase1 = dict(
    trackProducers = [
        'initialStepTracks',
        'highPtTripletStepTracks',
        'jetCoreRegionalStepTracks',
        'lowPtQuadStepTracks',
        'lowPtTripletStepTracks',
        'detachedQuadStepTracks',
        'detachedTripletStepTracks',
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
        "detachedTripletStep",
        "mixedTripletStep",
        "pixelLessStep",
        "tobTecStep"
    ],
)
trackingPhase1.toModify(earlyGeneralTracks, **_forPhase1)
trackingPhase1QuadProp.toModify(earlyGeneralTracks, **_forPhase1)

# For Phase1PU70
from RecoTracker.FinalTrackSelectors.trackListMerger_cfi import trackListMerger as _trackListMerger
from Configuration.Eras.Modifier_trackingPhase1PU70_cff import trackingPhase1PU70
trackingPhase1PU70.toReplaceWith(earlyGeneralTracks, _trackListMerger.clone(
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
# For Phase2PU140
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toReplaceWith(earlyGeneralTracks, _trackListMerger.clone(
    TrackProducers =['initialStepTracks',
                     'highPtTripletStepTracks',
                     'lowPtQuadStepTracks',
                     'lowPtTripletStepTracks',
                     'detachedQuadStepTracks',
                    ],
    hasSelector = [1,1,1,1,1],
    indivShareFrac = [1.0,0.16,0.095,0.09,0.09,0.09],
    selectedTrackQuals = cms.VInputTag(cms.InputTag("initialStepSelector","initialStep"),
                                       cms.InputTag("highPtTripletStepSelector","highPtTripletStep"),
                                       cms.InputTag("lowPtQuadStepSelector","lowPtQuadStep"),
                                       cms.InputTag("lowPtTripletStepSelector","lowPtTripletStep"),
                                       cms.InputTag("detachedQuadStep"),
                                       ),
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2,3,4), pQual=cms.bool(True) )
                             ),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
    )
)
