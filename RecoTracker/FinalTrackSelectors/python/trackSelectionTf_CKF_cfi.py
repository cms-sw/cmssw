from PhysicsTools.TensorFlow.tfGraphDefProducer_cfi import tfGraphDefProducer as _tfGraphDefProducer

trackSelectionTf_CKF = _tfGraphDefProducer.clone(
    ComponentName = "trackSelectionTf_CKF",
    FileName = "RecoTracker/FinalTrackSelectors/data/TrackTfClassifier/CKF_Run3_12_5_0_pre5.pb"
)

