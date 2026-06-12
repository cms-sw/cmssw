// Author: Felice Pantaleo - CERN
// Date: 03/2026
// A compact, read-only graph representation of the truth information in an event.
// The graph is built in the TruthGraphProducer module, which also fills the node metadata and associations.
// The graph is intended to be a common data format for various use cases (e.g. validation, analysis, visualization).

#ifndef PhysicsTools_TruthInfo_interface_TruthGraph_h
#define PhysicsTools_TruthInfo_interface_TruthGraph_h

#include <cstdint>
#include <span>
#include <vector>

class TruthGraph {
public:
  enum class NodeKind : uint8_t {
    GenEvent = 0,
    GenVertex = 1,
    GenParticle = 2,
    SimVertex = 3,
    SimTrack = 4,
  };

  // Edge categories (for visualization / filtering)
  enum class EdgeKind : uint8_t {
    Gen = 0,       // within GEN realm
    Sim = 1,       // within SIM realm
    GenToSim = 2,  // realm boundary GEN -> SIM
    SimToGen = 3   // reserved (we don't produce these now)
  };

  struct NodeRef {
    NodeKind kind = NodeKind::GenParticle;
    int64_t key = 0;  // GenParticle: index; SimTrack: trackId; SimVertex: index; GenVertex: barcode/index
  };

  TruthGraph() = default;

  // CSR out-edges: offsets.size() == nNodes+1
  // edges.size() == nEdges
  // edgeKind.size() == nEdges
  std::vector<uint32_t> offsets;
  std::vector<uint32_t> edges;
  std::vector<uint8_t> edgeKind;  // stores TruthGraph::EdgeKind as uint8_t

  // Node metadata: nodes.size() == nNodes
  std::vector<NodeRef> nodes;

  // Cached payload (optional)
  std::vector<int32_t> pdgId;         // 0 if not applicable
  std::vector<int16_t> status;        // 0 if not applicable
  std::vector<uint16_t> statusFlags;  // packed reco::GenStatusFlags, 0 if not available
  // Packed EncodedEventId for SIM nodes; 0 for GEN nodes
  std::vector<uint64_t> eventId;

  std::vector<int32_t> genEventOfNode;  // -1 for SIM; for GEN nodes = component id

  // Associations (nodeId -> nodeId). Only meaningful for SimTrack nodes.
  // -1 means "no association".
  std::vector<int32_t> simTrackToGen;  // SimTrack nodeId -> GenParticle nodeId
  std::vector<int32_t> simTrackToVtx;  // SimTrack nodeId -> SimVertex nodeId

  // SimVertex nodeId -> GenVertex nodeId provenance association, -1 if none.
  // Derived from primary SimTracks: a SimTrack's production SimVertex corresponds
  // to the production GenVertex of its associated GenParticle. Only meaningful for
  // SimVertex nodes.
  std::vector<int32_t> simVtxToGen;

  uint32_t nNodes() const { return static_cast<uint32_t>(nodes.size()); }
  uint32_t nEdges() const { return static_cast<uint32_t>(edges.size()); }

  uint32_t edgeBegin(uint32_t nodeId) const { return offsets.at(nodeId); }
  uint32_t edgeEnd(uint32_t nodeId) const { return offsets.at(nodeId + 1); }

  std::span<const uint32_t> children(uint32_t nodeId) const {
    const auto b = edgeBegin(nodeId);
    const auto e = edgeEnd(nodeId);
    return std::span<const uint32_t>(edges.data() + b, e - b);
  }

  std::span<const uint8_t> childrenEdgeKinds(uint32_t nodeId) const {
    const auto b = edgeBegin(nodeId);
    const auto e = edgeEnd(nodeId);
    return std::span<const uint8_t>(edgeKind.data() + b, e - b);
  }

  const NodeRef& nodeRef(uint32_t nodeId) const { return nodes.at(nodeId); }

  int32_t nodePdgId(uint32_t nodeId) const { return (nodeId < pdgId.size()) ? pdgId[nodeId] : 0; }

  int16_t nodeStatus(uint32_t nodeId) const { return (nodeId < status.size()) ? status[nodeId] : 0; }
  uint16_t nodeStatusFlags(uint32_t nodeId) const { return (nodeId < statusFlags.size()) ? statusFlags[nodeId] : 0; }
  uint64_t nodeEventId(uint32_t nodeId) const { return (nodeId < eventId.size()) ? eventId[nodeId] : 0ull; }

  int32_t nodeSimTrackToGen(uint32_t nodeId) const {
    return (nodeId < simTrackToGen.size()) ? simTrackToGen[nodeId] : -1;
  }

  int32_t nodeSimTrackToVtx(uint32_t nodeId) const {
    return (nodeId < simTrackToVtx.size()) ? simTrackToVtx[nodeId] : -1;
  }

  int32_t nodeSimVtxToGen(uint32_t nodeId) const { return (nodeId < simVtxToGen.size()) ? simVtxToGen[nodeId] : -1; }

  bool isConsistent() const;
};

#endif
