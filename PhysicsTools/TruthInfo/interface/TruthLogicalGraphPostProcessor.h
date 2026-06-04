#ifndef PhysicsTools_TruthInfo_interface_TruthLogicalGraphPostProcessor_h
#define PhysicsTools_TruthInfo_interface_TruthLogicalGraphPostProcessor_h

#include <cstdint>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "PhysicsTools/TruthInfo/interface/Graph.h"

namespace truth {

  struct LogicalGraphPostProcessingConfig {
    bool collapseIntermediateGenParticles = true;

    // If empty, no seed-based graph cut is applied.
    std::vector<int32_t> seedPdgIds;

    // For each selected seed particle, keep this many generations of parents
    // above the seed before keeping the full downstream graph.
    uint32_t seedParentDepth = 0;

    // Particles with these exact PDG ids are removed from the final logical graph.
    // If such a particle is internal, its production and decay vertices are merged
    // so that the graph remains navigable.
    std::vector<int32_t> ignoredPdgIds;

    // Exact logical particle ids to remove from the final logical graph.
    // These ids refer to the graph state at the moment the ignored-particle
    // collapsing step is applied.
    std::vector<uint32_t> ignoredParticleIds;

    // If true, post-processing is allowed to merge a GEN-only vertex and a
    // SIM-only vertex when they are connected to the same visible particle and
    // their four-positions are compatible within genSimVertexPositionTolerance.
    //
    // This is intentionally done at logical-graph level: the raw graph can still
    // keep GenVertex and SimVertex as distinct provenance objects.
    bool mergeGenSimVerticesByPosition = true;

    // Absolute tolerance used for each x, y, z, t component when matching
    // GEN-only and SIM-only vertices by position.
    double genSimVertexPositionTolerance = 1e-6;
  };

  class TruthLogicalGraphPostProcessor {
  public:
    TruthLogicalGraphPostProcessor() = default;
    explicit TruthLogicalGraphPostProcessor(LogicalGraphPostProcessingConfig config);

    static edm::ParameterSetDescription psetDescription();
    static LogicalGraphPostProcessingConfig configFromPSet(edm::ParameterSet const& pset);

    [[nodiscard]] Graph process(Graph input) const;

  private:
    LogicalGraphPostProcessingConfig config_;
  };

}  // namespace truth

#endif
