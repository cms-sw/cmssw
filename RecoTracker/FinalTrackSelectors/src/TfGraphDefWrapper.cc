#include "RecoTracker/FinalTrackSelectors/interface/TfGraphDefWrapper.h"

TfGraphDefWrapper::TfGraphDefWrapper(tensorflow::GraphDef* graph) : graphDef_(graph) {}

const tensorflow::GraphDef* TfGraphDefWrapper::getGraphDef() const { return graphDef_.get(); }
