#ifndef PhysicsTools_TruthInfo_interface_Graph_h
#define PhysicsTools_TruthInfo_interface_Graph_h

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "DataFormats/Math/interface/LorentzVector.h"

namespace truth {

  struct Checkpoint {
    uint32_t checkpointId = 0;
    math::XYZTLorentzVectorF position;
    math::XYZTLorentzVectorF momentum;
  };

  struct ParticleData {
    // Optional provenance/debug back-references to the raw TruthGraph nodes.
    // -1 means "not available".
    int32_t genNode = -1;
    int32_t simNode = -1;

    // Merged metadata.
    int32_t pdgId = 0;
    int16_t status = 0;

    // SIM event id when available, 0 otherwise.
    uint64_t eventId = 0;

    // GEN connected component id from the raw TruthGraph, -1 if not applicable.
    int32_t genEvent = -1;

    // Standalone payload.
    // Convention: "best available" momentum.
    // Prefer SIM if present, otherwise GEN, otherwise default-constructed.
    math::XYZTLorentzVectorD momentum;

    // Optional trajectory checkpoints.
    std::vector<Checkpoint> checkpoints;

    [[nodiscard]] bool hasGen() const { return genNode >= 0; }
    [[nodiscard]] bool hasSim() const { return simNode >= 0; }
    [[nodiscard]] bool valid() const { return hasGen() || hasSim(); }
  };

  struct VertexData {
    // Optional provenance/debug back-references to the raw TruthGraph nodes.
    // -1 means "not available".
    int32_t genNode = -1;
    int32_t simNode = -1;

    // SIM event id when available, 0 otherwise.
    uint64_t eventId = 0;

    // GEN connected component id from the raw TruthGraph, -1 if not applicable.
    int32_t genEvent = -1;

    // Standalone payload.
    // Convention: "best available" position.
    // Prefer SIM if present, otherwise GEN, otherwise default-constructed.
    math::XYZTLorentzVectorD position;

    [[nodiscard]] bool hasGen() const { return genNode >= 0; }
    [[nodiscard]] bool hasSim() const { return simNode >= 0; }
    [[nodiscard]] bool valid() const { return hasGen() || hasSim(); }
  };

  class Graph;
  class Particle;
  class Vertex;

  class Particle {
  public:
    Particle() = default;
    Particle(Graph const* graph, uint32_t id) : graph_(graph), id_(id) {}

    [[nodiscard]] bool valid() const { return graph_ != nullptr; }
    [[nodiscard]] uint32_t id() const { return id_; }

    [[nodiscard]] const ParticleData& data() const;

    [[nodiscard]] bool hasGen() const;
    [[nodiscard]] bool hasSim() const;
    [[nodiscard]] int32_t pdgId() const;
    [[nodiscard]] int16_t status() const;
    [[nodiscard]] uint64_t eventId() const;
    [[nodiscard]] int32_t genEvent() const;
    [[nodiscard]] const math::XYZTLorentzVectorD& momentum() const;

    [[nodiscard]] std::span<const Checkpoint> checkpoints() const;
    [[nodiscard]] bool hasCheckpoints() const;
    [[nodiscard]] std::optional<Checkpoint> checkpoint(uint32_t checkpointId) const;

    [[nodiscard]] bool isRoot() const;
    [[nodiscard]] bool isLeaf() const;

    [[nodiscard]] std::vector<Vertex> productionVertices() const;
    [[nodiscard]] std::vector<Vertex> decayVertices() const;

    [[nodiscard]] std::vector<Particle> parents() const;
    [[nodiscard]] std::vector<Particle> children() const;

    [[nodiscard]] std::vector<Particle> ancestors() const;
    [[nodiscard]] std::vector<Particle> descendants() const;

    [[nodiscard]] bool hasAncestorPdgId(int pdgId) const;
    [[nodiscard]] std::optional<Particle> firstAncestorWithPdgId(int pdgId) const;
    [[nodiscard]] std::optional<Particle> firstCommonAncestor(Particle other) const;

    [[nodiscard]] bool operator==(Particle const& other) const {
      return graph_ == other.graph_ && id_ == other.id_;
    }
    [[nodiscard]] bool operator!=(Particle const& other) const { return !(*this == other); }

  private:
    Graph const* graph_ = nullptr;
    uint32_t id_ = 0;
  };

  class Vertex {
  public:
    Vertex() = default;
    Vertex(Graph const* graph, uint32_t id) : graph_(graph), id_(id) {}

    [[nodiscard]] bool valid() const { return graph_ != nullptr; }
    [[nodiscard]] uint32_t id() const { return id_; }

    [[nodiscard]] const VertexData& data() const;

    [[nodiscard]] bool hasGen() const;
    [[nodiscard]] bool hasSim() const;
    [[nodiscard]] uint64_t eventId() const;
    [[nodiscard]] int32_t genEvent() const;
    [[nodiscard]] const math::XYZTLorentzVectorD& position() const;

    [[nodiscard]] bool isSource() const;
    [[nodiscard]] bool isSink() const;

    [[nodiscard]] std::vector<Particle> incomingParticles() const;
    [[nodiscard]] std::vector<Particle> outgoingParticles() const;

    [[nodiscard]] bool operator==(Vertex const& other) const {
      return graph_ == other.graph_ && id_ == other.id_;
    }
    [[nodiscard]] bool operator!=(Vertex const& other) const { return !(*this == other); }

  private:
    Graph const* graph_ = nullptr;
    uint32_t id_ = 0;
  };

  class Graph {
  public:
    using size_type = uint32_t;

    std::vector<ParticleData> particles;
    std::vector<VertexData> vertices;

    // Particle -> decay vertices
    std::vector<uint32_t> particleToDecayVertexOffsets;
    std::vector<uint32_t> particleToDecayVertices;

    // Particle -> production vertices
    std::vector<uint32_t> particleToProductionVertexOffsets;
    std::vector<uint32_t> particleToProductionVertices;

    // Vertex -> outgoing particles
    std::vector<uint32_t> vertexToOutgoingParticleOffsets;
    std::vector<uint32_t> vertexToOutgoingParticles;

    // Vertex -> incoming particles
    std::vector<uint32_t> vertexToIncomingParticleOffsets;
    std::vector<uint32_t> vertexToIncomingParticles;

    [[nodiscard]] size_type nParticles() const { return static_cast<size_type>(particles.size()); }
    [[nodiscard]] size_type nVertices() const { return static_cast<size_type>(vertices.size()); }

    [[nodiscard]] bool empty() const { return particles.empty() && vertices.empty(); }

    [[nodiscard]] Particle particle(size_type id) const;
    [[nodiscard]] Vertex vertex(size_type id) const;

    [[nodiscard]] std::vector<Particle> particleViews() const;
    [[nodiscard]] std::vector<Vertex> vertexViews() const;

    [[nodiscard]] std::vector<Particle> roots() const;
    [[nodiscard]] std::vector<Particle> leaves() const;

    [[nodiscard]] std::vector<Vertex> sourceVertices() const;
    [[nodiscard]] std::vector<Vertex> sinkVertices() const;

    [[nodiscard]] std::span<const uint32_t> decayVertices(size_type particleId) const {
      const auto b = particleToDecayVertexOffsets.at(particleId);
      const auto e = particleToDecayVertexOffsets.at(particleId + 1);
      return std::span<const uint32_t>(particleToDecayVertices.data() + b, e - b);
    }

    [[nodiscard]] std::span<const uint32_t> productionVertices(size_type particleId) const {
      const auto b = particleToProductionVertexOffsets.at(particleId);
      const auto e = particleToProductionVertexOffsets.at(particleId + 1);
      return std::span<const uint32_t>(particleToProductionVertices.data() + b, e - b);
    }

    [[nodiscard]] std::span<const uint32_t> outgoingParticles(size_type vertexId) const {
      const auto b = vertexToOutgoingParticleOffsets.at(vertexId);
      const auto e = vertexToOutgoingParticleOffsets.at(vertexId + 1);
      return std::span<const uint32_t>(vertexToOutgoingParticles.data() + b, e - b);
    }

    [[nodiscard]] std::span<const uint32_t> incomingParticles(size_type vertexId) const {
      const auto b = vertexToIncomingParticleOffsets.at(vertexId);
      const auto e = vertexToIncomingParticleOffsets.at(vertexId + 1);
      return std::span<const uint32_t>(vertexToIncomingParticles.data() + b, e - b);
    }

    [[nodiscard]] bool isConsistent() const;

  private:
    friend class Particle;
    friend class Vertex;

    [[nodiscard]] std::vector<Vertex> productionVerticesOf(size_type particleId) const;
    [[nodiscard]] std::vector<Vertex> decayVerticesOf(size_type particleId) const;

    [[nodiscard]] std::vector<Particle> parentsOf(size_type particleId) const;
    [[nodiscard]] std::vector<Particle> childrenOf(size_type particleId) const;

    [[nodiscard]] std::vector<Particle> ancestorsOf(size_type particleId) const;
    [[nodiscard]] std::vector<Particle> descendantsOf(size_type particleId) const;

    [[nodiscard]] std::optional<Particle> firstAncestorWithPdgIdOf(size_type particleId, int pdgId) const;
    [[nodiscard]] std::optional<Particle> firstCommonAncestorOf(size_type a, size_type b) const;

    [[nodiscard]] std::vector<Particle> incomingParticlesOf(size_type vertexId) const;
    [[nodiscard]] std::vector<Particle> outgoingParticlesOf(size_type vertexId) const;
  };

}  // namespace truth

#endif