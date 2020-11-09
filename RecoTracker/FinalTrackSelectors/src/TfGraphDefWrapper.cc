#include "RecoTracker/FinalTrackSelectors/interface/TfGraphDefWrapper.h"

TfGraphDefWrapper::TfGraphDefWrapper(const std::unique_ptr<tensorflow::GraphDef> graph) { graphDef_ = graph; }

const std::unique_ptr<tensorflow::GraphDef> TfGraphDefWrapper::getGraphDef() const { return graphDef_; }
