from PhysicsTools.TensorFlow.tfGraphDefProducer_cfi import tfGraphDefProducer as _tfGraphDefProducer
trackSelectionTf = _tfGraphDefProducer.clone(
    ComponentName = "trackSelectionTf",
    FileName = "RecoTracker/FinalTrackSelectors/data/TrackTfClassifier/MkFit4plus3_2021Run3.pb"
)


