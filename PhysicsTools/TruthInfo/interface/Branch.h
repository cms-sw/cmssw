// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#ifndef PhysicsTools_TruthInfo_interface_Branch_h
#define PhysicsTools_TruthInfo_interface_Branch_h

#include <cstdint>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "DataFormats/Math/interface/LorentzVector.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"
#include "SimDataFormats/TruthInfo/interface/ParticleData.h"

namespace truth {

  // How far below the root(s) a Branch extends.
  enum class ClosureKind : uint8_t { Subtree, StableLeaves, DepthN, UntilPdgId, UntilLevels, Predicate };

  struct ClosureSpec {
    ClosureKind kind = ClosureKind::Subtree;
    uint32_t maxDepth = 0;                 // DepthN: generations kept below each root (0 = roots only)
    std::vector<int32_t> stopPdgIds;       // UntilPdgId: stop at (and include) particles with these ids (excl. root)
    std::function<bool(Particle)> stopAt;  // Predicate: stop at (and include) particles where true (incl. root)
    uint32_t levelFlags = 0;  // UntilLevels: stop at (and include) particles at any of these levels (incl. root)
                              // Note: levelFlags==0 is equivalent to the full branch

    static ClosureSpec subtree() { return {}; }
    static ClosureSpec stableLeaves() { return {ClosureKind::StableLeaves, 0, {}, {}, 0}; }
    static ClosureSpec depth(uint32_t n) { return {ClosureKind::DepthN, n, {}, {}, 0}; }
    static ClosureSpec untilPdgId(std::vector<int32_t> ids) {
      return {ClosureKind::UntilPdgId, 0, std::move(ids), {}, 0};
    }
    static ClosureSpec untilLevel(LevelFlag level) {
      return {ClosureKind::UntilLevels, 0, {}, {}, static_cast<uint32_t>(level)};
    }
    static ClosureSpec untilLevels(std::vector<LevelFlag> const& levels) {
      uint32_t flag = 0;
      for (auto level : levels)
        flag |= static_cast<uint32_t>(level);
      return {ClosureKind::UntilLevels, 0, {}, {}, flag};
    }
    static ClosureSpec predicate(std::function<bool(Particle)> p) {
      return {ClosureKind::Predicate, 0, {}, std::move(p), 0};
    }
  };

  // A Branch is a lightweight, non-owning view of a coherent subgraph: one or
  // more root particles plus a closure of their descendants. Members are
  // recomputed on demand from the Graph; the Branch stores no graph data and is
  // not an EDM product. It is the truth-side object that reco objects are matched
  // to, the natural successor to the static CaloParticle/TrackingParticle.
  class Branch {
  public:
    Branch() = delete;
    Branch(Graph const* graph, uint32_t rootId, ClosureSpec spec = ClosureSpec::subtree());
    Branch(Graph const* graph, std::vector<uint32_t> rootIds, ClosureSpec spec = ClosureSpec::subtree());
    Branch(Particle const* particle, ClosureSpec spec = ClosureSpec::subtree());

    [[nodiscard]] Graph const* graph() const { return graph_; }
    [[nodiscard]] Particle root() const;
    [[nodiscard]] std::vector<Particle> roots() const;
    [[nodiscard]] std::vector<uint32_t> rootIds() const { return roots_; }
    [[nodiscard]] ClosureSpec const& closure() const { return spec_; }

    // Roots and descendants up to and including the particles where the closure stops, ascending particle id.
    [[nodiscard]] std::vector<uint32_t> memberIds() const;
    [[nodiscard]] std::vector<Particle> members() const;

    // Only the members that meet the closure condition, ascending particle id.
    [[nodiscard]] std::vector<Particle> closureLeaves() const;

    // Members that are stable leaves (up to and including the closure), ascending particle id.
    [[nodiscard]] std::vector<Particle> stableLeaves() const;

    // The members no other member covers: the final-state leaves of a full subtree,
    // or the particles the closure stopped at when it truncates.
    [[nodiscard]] std::vector<uint32_t> leaves() const;

    // Kinematics, summed over the "frontier" leaves, so a truncated closure counts the particle
    // it stopped at and never counts a particle together with its own ancestor.
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
    // Anything that is not the signal interaction. NOT bunchCrossing() != 0: in-time
    // pileup carries bunch crossing 0 and a nonzero event number, and the default
    // production keeps in-time pileup only.
    [[nodiscard]] bool isFromPileup() const { return !isSignal(); }
    [[nodiscard]] bool isSignal() const { return bunchCrossing() == 0 && event() == 0; }

    // Relations between branches.
    [[nodiscard]] std::optional<Particle> commonAncestor(Branch const& other) const;
    [[nodiscard]] Branch merged(Branch const& other) const;

  private:
    void validate();
    // Fills stopIds (when non-null) with the particles where the closure stops
    [[nodiscard]] std::vector<uint32_t> traverse(std::vector<uint32_t>* stopIds = nullptr) const;

    Graph const* graph_;
    std::vector<uint32_t> roots_;
    ClosureSpec spec_;
  };

}  // namespace truth

#endif
