#ifndef TrackTfGraph_TfGraphDefWrapper_h
#define TrackTfGraph_TfGraphDefWrapper_h

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

class TfGraphDefWrapper {
public:
  TfGraphDefWrapper(tensorflow::GraphDef*);
  tensorflow::GraphDef* GetGraphDef() const;

private:
  tensorflow::GraphDef* graphDef_;
};

#endif
