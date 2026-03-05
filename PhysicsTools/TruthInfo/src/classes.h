#ifndef PhysicsTools_TruthInfo_src_classes_h
#define PhysicsTools_TruthInfo_src_classes_h

#include "PhysicsTools/TruthInfo/interface/TruthGraph.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  struct dictionary {
    TruthGraph graph;
    edm::Wrapper<TruthGraph> wgraph;
  };
}  // namespace

#endif
