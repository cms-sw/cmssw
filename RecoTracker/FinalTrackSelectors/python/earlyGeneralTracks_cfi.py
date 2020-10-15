import FWCore.ParameterSet.Config as cms
from RecoTracker.FinalTrackSelectors.TrackCollectionMerger_cfi import *
from RecoTracker.FinalTrackSelectors.trackAlgoPriorityOrder_cfi import trackAlgoPriorityOrder

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
earlyGeneralTracks =  TrackCollectionMerger.clone(
    trackProducers = ['initialStepTracks',
                      'jetCoreRegionalStepTracks',
                      'lowPtTripletStepTracks',
                      'pixelPairStepTracks',
                      'detachedTripletStepTracks',
                      'mixedTripletStepTracks',
                      'pixelLessStepTracks',
                      'tobTecStepTracks'
                      ],
    inputClassifiers =['initialStep',
                       'jetCoreRegionalStep',
                       'lowPtTripletStep',
                       'pixelPairStep',
                       'detachedTripletStep',
                       'mixedTripletStep',
                       'pixelLessStep',
                       'tobTecStep'
                       ]
)
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
        'initialStepSelector',
        'lowPtTripletStepSelector',
        'pixelPairStepSelector',
        'detachedTripletStep',
        'mixedTripletStep',
        'pixelLessStepSelector',
        'tobTecStep'
    ]
)
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
_forPhase1 = dict(
    trackProducers = [
        'initialStepTracks',
        'highPtTripletStepTracks',
        'jetCoreRegionalStepTracks',
        'lowPtQuadStepTracks',
        'lowPtTripletStepTracks',
        'detachedQuadStepTracks',
        'detachedTripletStepTracks',
        'pixelPairStepTracks',
        'mixedTripletStepTracks',
        'pixelLessStepTracks',
        'tobTecStepTracks'
    ],
    inputClassifiers = [
        'initialStep',
        'highPtTripletStep',
        'jetCoreRegionalStep',
        'lowPtQuadStep',
        'lowPtTripletStep',
        'detachedQuadStep',
        'detachedTripletStep',
        'pixelPairStep',
        'mixedTripletStep',
        'pixelLessStep',
        'tobTecStep'
    ],
)
trackingPhase1.toModify(earlyGeneralTracks, **_forPhase1)

# For Phase2PU140
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
from RecoTracker.FinalTrackSelectors.trackListMerger_cfi import trackListMerger as _trackListMerger
trackingPhase2PU140.toReplaceWith(earlyGeneralTracks, _trackListMerger.clone(
    TrackProducers =['initialStepTracks',
                     'highPtTripletStepTracks',
                     'lowPtQuadStepTracks',
                     'lowPtTripletStepTracks',
                     'detachedQuadStepTracks',
                     'pixelPairStepTracks',
                    ],
    hasSelector = [1,1,1,1,1,1],
    indivShareFrac = [1.0,0.16,0.095,0.09,0.09,0.09],
    selectedTrackQuals = ['initialStepSelector:initialStep',
                          'highPtTripletStepSelector:highPtTripletStep',
                          'lowPtQuadStepSelector:lowPtQuadStep',
                          'lowPtTripletStepSelector:lowPtTripletStep',
                          'detachedQuadStep',
                          'pixelPairStepSelector:pixelPairStep',
                          ],
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2,3,4,5), pQual=cms.bool(True) ) 
	),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
    )
)
from Configuration.ProcessModifiers.vectorHits_cff import vectorHits
def _extend_pixelLess(x):
    x.TrackProducers += ['pixelLessStepTracks']
    x.hasSelector += [1]
    x.indivShareFrac += [0.095]
    x.selectedTrackQuals += ['pixelLessStepSelector:pixelLessStep']
    x.setsToMerge[0].tLists += [6]
(trackingPhase2PU140 & vectorHits).toModify(earlyGeneralTracks, _extend_pixelLess)

