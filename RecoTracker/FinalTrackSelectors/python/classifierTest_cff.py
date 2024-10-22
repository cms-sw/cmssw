from RecoTracker.FinalTrackSelectors.TrackCutClassifier_cff import *
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierDetached_cfi import *

testTrackClassifier1 = TrackMVAClassifierPrompt.clone(
    src = 'initialStepTracks',
    mva = dict(GBRForestLabel = 'MVASelectorIter0_13TeV'),
    qualityCuts = [-0.9,-0.8,-0.7]
)

testTrackClassifier2 = TrackCutClassifier.clone(
    src = 'initialStepTracks',
    mva = dict(minPixelHits = [0,1,1])
)
    
from RecoTracker.FinalTrackSelectors.ClassifierMerger_cfi import *
testMergedClassifier = ClassifierMerger.clone(
    inputClassifiers=['testTrackClassifier1','testTrackClassifier2']
)

from RecoTracker.FinalTrackSelectors.TrackCollectionMerger_cfi import *
testTrackMerger = TrackCollectionMerger.clone(
    trackProducers = ['initialStepTracks'],
    inputClassifiers =['testMergedClassifier'],
    minQuality = 'tight'
)

testTrackClassifier3 = TrackMVAClassifierDetached.clone(
    src = 'detachedTripletStepTracks',
    mva = dict(GBRForestLabel = 'MVASelectorIter3_13TeV'),
    qualityCuts = [-0.5,0.0,0.5]
)

from RecoTracker.FinalTrackSelectors.trackAlgoPriorityOrder_cfi import trackAlgoPriorityOrder
from RecoTracker.FinalTrackSelectors.TrackCollectionMerger_cfi import *
testTrackMerger2 = TrackCollectionMerger.clone(
    trackProducers = ['initialStepTracks','detachedTripletStepTracks'],
    inputClassifiers =['testMergedClassifier','testTrackClassifier3'],
    minQuality = 'tight'
)

testTrackCloning = cms.Sequence(testTrackClassifier1*testTrackClassifier2*testTrackClassifier3*
                                testMergedClassifier*testTrackMerger*testTrackMerger2)
