from PhysicsTools.TensorFlow.tfGraphDefProducer_cfi import tfGraphDefProducer as _tfGraphDefProducer

trackSelectionTf_CKF = _tfGraphDefProducer.clone(
    ComponentName = "trackSelectionTf_CKF",
    FileName = "RecoTracker/FinalTrackSelectors/data/TrackTfClassifier/CKF_2021Run3.pb"
)

