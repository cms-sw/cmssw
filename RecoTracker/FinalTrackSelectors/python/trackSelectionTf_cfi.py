from DataFormats.TrackTfGraph.tfGraphDefProducer_cfi import tfGraphDefProducer as _tfGraphDefProducer
trackSelectionTf = _tfGraphDefProducer.clone(
    ComponentName = "trackSelectionTf",
    FileName = "RecoTracker/FinalTrackSelectors/data/frozen_graph.pb"
)
