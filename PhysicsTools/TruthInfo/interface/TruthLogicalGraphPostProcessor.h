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
