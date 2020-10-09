#include "RecoTracker/FinalTrackSelectors/interface/TfGraphDefWrapper.h"

TfGraphDefWrapper::TfGraphDefWrapper(tensorflow::GraphDef* graph) { graphDef_ = graph; }

tensorflow::GraphDef* TfGraphDefWrapper::GetGraphDef() const { return graphDef_; }
