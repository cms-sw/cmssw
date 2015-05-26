from RecoTracker.FinalTrackSelectors.TrackCutClassifier_cfi import *

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


testTrackCloning = cms.Sequence(testTrackClassifier1*testTrackClassifier2*testMergedClassifier*testTrackMerger)
