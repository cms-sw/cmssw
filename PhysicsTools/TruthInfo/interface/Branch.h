// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#ifndef PhysicsTools_TruthInfo_interface_Branch_h
#define PhysicsTools_TruthInfo_interface_Branch_h

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include "DataFormats/Math/interface/LorentzVector.h"
#include "PhysicsTools/TruthInfo/interface/Graph.h"

namespace truth {

  // How far below the root(s) a Branch extends.
  enum class ClosureKind : uint8_t { Subtree, StableLeaves, DepthN, UntilPdgId, Predicate };

  struct ClosureSpec {
    ClosureKind kind = ClosureKind::Subtree;
    uint32_t maxDepth = 0;                 // DepthN: generations kept below each root (0 = roots only)
    std::vector<int32_t> stopPdgIds;       // UntilPdgId: stop at (and include) particles with these ids
    std::function<bool(Particle)> stopAt;  // Predicate: stop at (and include) particles where true

    static ClosureSpec subtree() { return {}; }
    static ClosureSpec stableLeaves() { return {ClosureKind::StableLeaves, 0, {}, {}}; }
    static ClosureSpec depth(uint32_t n) { return {ClosureKind::DepthN, n, {}, {}}; }
    static ClosureSpec untilPdgId(std::vector<int32_t> ids) { return {ClosureKind::UntilPdgId, 0, std::move(ids), {}}; }
    static ClosureSpec predicate(std::function<bool(Particle)> p) {
      return {ClosureKind::Predicate, 0, {}, std::move(p)};
    }
  };

  // A Branch is a lightweight, non-owning view of a coherent subgraph: one or
  // more root particles plus a closure of their descendants. Members are
  // recomputed on demand from the Graph; the Branch stores no graph data and is
  // not an EDM product. It is the truth-side object that reco objects are matched
  // to, the natural successor to the static CaloParticle/TrackingParticle.
  class Branch {
  public:
    Branch() = default;
    Branch(Graph const* graph, uint32_t rootId, ClosureSpec spec = ClosureSpec::subtree());
    Branch(Graph const* graph, std::vector<uint32_t> rootIds, ClosureSpec spec = ClosureSpec::subtree());

    [[nodiscard]] bool valid() const { return graph_ != nullptr && !roots_.empty(); }
    [[nodiscard]] Graph const* graph() const { return graph_; }
    [[nodiscard]] Particle root() const;
    [[nodiscard]] std::vector<Particle> roots() const;
    [[nodiscard]] std::vector<uint32_t> rootIds() const { return roots_; }
    [[nodiscard]] ClosureSpec const& closure() const { return spec_; }

    // Closure members (roots + selected descendants), ascending particle id.
    [[nodiscard]] std::vector<uint32_t> memberIds() const;
    [[nodiscard]] std::vector<Particle> members() const;
    [[nodiscard]] std::vector<Particle> stableLeaves() const;

    // Kinematics, summed over the stable final-state leaves.
    [[nodiscard]] math::XYZTLorentzVectorD p4() const;
    [[nodiscard]] math::XYZTLorentzVectorD visibleP4() const;  // excludes neutrinos
    [[nodiscard]] double energy() const { return p4().energy(); }
    [[nodiscard]] double visibleEnergy() const { return visibleP4().energy(); }
    [[nodiscard]] double invisibleEnergy() const;

    // Tagging / origin.
    [[nodiscard]] int32_t rootPdgId() const;
    [[nodiscard]] std::optional<Particle> originWithPdgId(int32_t pdgId) const;
    [[nodiscard]] bool hasHeavyFlavor(int32_t quarkFlavor) const;  // any member is a flavor-q hadron

    // Provenance (pile-up aware): the source event of the root.
    [[nodiscard]] int32_t genEvent() const;
    [[nodiscard]] int bunchCrossing() const;
    [[nodiscard]] int event() const;
    [[nodiscard]] bool isInTime() const { return bunchCrossing() == 0; }
    [[nodiscard]] bool isFromPileup() const { return bunchCrossing() != 0; }
    [[nodiscard]] bool isSignal() const { return bunchCrossing() == 0 && event() == 0; }

    // Relations between branches.
    [[nodiscard]] std::optional<Particle> commonAncestor(Branch const& other) const;
    [[nodiscard]] Branch merged(Branch const& other) const;

  private:
    [[nodiscard]] std::vector<uint32_t> traverse() const;

    Graph const* graph_ = nullptr;
    std::vector<uint32_t> roots_;
    ClosureSpec spec_;
  };

}  // namespace truth

#endif
