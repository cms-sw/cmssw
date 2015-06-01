from RecoTracker.FinalTrackSelectors.TrackCutClassifier_cfi import *
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierDetached_cfi import *



testTrackClassifier1 = TrackCutClassifier.clone(
    src = 'initialStepTracks'
    )

testTrackClassifier2 = TrackCutClassifier.clone()
testTrackClassifier2.src = 'initialStepTracks'
testTrackClassifier2.mva.minPixelHits = [0,1,1]

    
from RecoTracker.FinalTrackSelectors.ClassifierMerger_cfi import *
testMergedClassifier = ClassifierMerger.clone()
testMergedClassifier.inputClassifiers=['testTrackClassifier1','testTrackClassifier2']

from RecoTracker.FinalTrackSelectors.TrackCollectionMerger_cfi import *
testTrackMerger = TrackCollectionMerger.clone()
testTrackMerger.trackProducers = ['initialStepTracks']
testTrackMerger.inputClassifiers =['testMergedClassifier']
testTrackMerger.minQuality = 'tight'

testTrackClassifier3 = TrackMVAClassifierDetached.clone()
testTrackClassifier3.src = 'detachedTripletStepTracks'
testTrackClassifier3.GBRForestLabel = 'MVASelectorIter3_13TeV'
testTrackClassifier3.qualityCuts = [-0.5,0.0,0.5]


from RecoTracker.FinalTrackSelectors.TrackCollectionMerger_cfi import *
testTrackMerger2 = TrackCollectionMerger.clone()
testTrackMerger2.trackProducers = ['initialStepTracks','detachedTripletStepTracks']
testTrackMerger2.inputClassifiers =['testMergedClassifier','testTrackClassifier3']
testTrackMerger2.minQuality = 'tight'





testTrackCloning = cms.Sequence(testTrackClassifier1*testTrackClassifier2*testTrackClassifier3*
                                testMergedClassifier*testTrackMerger*testTrackMerger2)
