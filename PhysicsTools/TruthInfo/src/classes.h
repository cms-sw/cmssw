// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#ifndef PhysicsTools_TruthInfo_src_classes_h
#define PhysicsTools_TruthInfo_src_classes_h

#include "DataFormats/Common/interface/Wrapper.h"
#include "PhysicsTools/TruthInfo/interface/Graph.h"
#include "PhysicsTools/TruthInfo/interface/TruthGraph.h"
#include "PhysicsTools/TruthInfo/interface/LogicalGraphHitIndex.h"

namespace {
  struct dictionary {
    TruthGraph rawTruthGraph;
    edm::Wrapper<TruthGraph> wrappedRawTruthGraph;

    TruthGraph::NodeRef rawTruthNodeRef;
    std::vector<TruthGraph::NodeRef> rawTruthNodeRefs;

    truth::Graph logicalTruthGraph;
    edm::Wrapper<truth::Graph> wrappedLogicalTruthGraph;

    truth::Checkpoint logicalTruthCheckpoint;
    std::vector<truth::Checkpoint> logicalTruthCheckpoints;

    truth::ParticleData logicalTruthParticleData;
    std::vector<truth::ParticleData> logicalTruthParticleDataVec;

    truth::VertexData logicalTruthVertexData;
    std::vector<truth::VertexData> logicalTruthVertexDataVec;

    truth::LogicalGraphHitIndex logicalGraphHitIndex;
    truth::LogicalGraphHitIndex::Hit logicalGraphHitIndexHit;
    std::vector<truth::LogicalGraphHitIndex::Hit> logicalGraphHitIndexHitVector;
    truth::LogicalGraphHitIndex::Channel logicalGraphHitIndexChannel;
    std::vector<truth::LogicalGraphHitIndex::Channel> logicalGraphHitIndexChannelVector;
  };
}  // namespace

#endif
