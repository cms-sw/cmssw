#ifndef RecoTracker_FinalTrackSelectors_TfGraphDefWrapper_h
#define RecoTracker_FinalTrackSelectors_TfGraphDefWrapper_h

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

class TfGraphDefWrapper {
public:
  TfGraphDefWrapper(const std::unique_ptr<tensorflow::GraphDef>);
  const std::unique_ptr<tensorflow::GraphDef> getGraphDef() const;

private:
  const std::unique_ptr<tensorflow::GraphDef> graphDef_;
  //tensorflow::GraphDef* graphDef_;
};

#endif
