from RecoTracker.FinalTrackSelectors.tfGraphDefProducer_cfi import tfGraphDefProducer as _tfGraphDefProducer
trackSelectionTf = _tfGraphDefProducer.clone(
    ComponentName = "trackSelectionTf",
    FileName = "RecoTracker/FinalTrackSelectors/data/QCDFlatPU_QCDHighPt_ZEE_DisplacedSUSY_2020.pb"
)
