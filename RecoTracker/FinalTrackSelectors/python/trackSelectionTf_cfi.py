from PhysicsTools.TensorFlow.tfGraphDefProducer_cfi import tfGraphDefProducer as _tfGraphDefProducer
trackSelectionTf = _tfGraphDefProducer.clone(
    ComponentName = "trackSelectionTf",
    FileName = "RecoTracker/FinalTrackSelectors/data/TrackTfClassifier/MkFit_Run3_12_5_0_pre5.pb"
)
trackSelectionTfPLess = _tfGraphDefProducer.clone(
    ComponentName = "trackSelectionTfPLess",
    FileName = "RecoTracker/FinalTrackSelectors/data/TrackTfClassifier/MkFitPixelLessOnly_Run3_12_5_0_pre5.pb"
)


