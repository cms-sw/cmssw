// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

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
    // The most upstream particle of each matching chain becomes a root of the
    // selected graph. The special value 0 disables the selection and keeps the
    // full graph (debugging escape hatch).
    std::vector<int32_t> seedPdgIds;

    // Seed on hadrons by heavy-flavor content instead of (or in addition to)
    // exact PDG ids: a particle whose PDG id is a hadron containing any of these
    // quark flavors becomes a seed. Use 5 for b hadrons, 4 for c hadrons. This
    // lets the user select e.g. all B-hadron decay subgraphs without listing
    // every B species. OR-ed with seedPdgIds.
    std::vector<int32_t> seedHadronFlavors;

    // For each selected root, keep this many generations of ancestors above it
    // as context only: the ancestors and connecting vertices are kept, but not
    // their other descendants.
    uint32_t seedParentDepth = 0;

    // If true (default), stable final-state GEN particles outside the selected
    // subgraph are kept and attached to an artificial UnderlyingEvent source
    // vertex. If false, they are dropped, giving a focused subgraph that
    // contains only the selection and its truncated upstream (ISR) context.
    // Only meaningful when a selection is active (seedPdgIds/decayPdgIdGroups).
    bool keepStableSpectators = true;

    // Decay patterns of interest. Each group is an unordered, charge-sensitive
    // multiset of PDG ids; groups are OR-ed.
    //
    // Without seedPdgIds: a vertex whose outgoing PDG ids contain a group as a
    // sub-multiset is selected, and the matched particles plus their downstream
    // subgraphs are kept.
    //
    // With seedPdgIds: only seed roots whose effective decay products (after
    // following same-PDG radiating copy chains) contain a group are kept. If
    // the event contains no particle with a seed PDG id at all, the direct
    // vertex search is used as a fallback.
    std::vector<std::vector<int32_t>> decayPdgIdGroups;

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
    // GEN-only and SIM-only vertices by position. Keep in sync with the
    // default in psetDescription().
    double genSimVertexPositionTolerance = 5e-3;
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
