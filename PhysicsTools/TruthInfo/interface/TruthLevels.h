// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
//
// Levels of the truth graph: the a-priori definition of WHAT a truth object is, for the
// truth-driven direction of the association.
//
// A level must be an ANTICHAIN: no member may be an ancestor of another. A nested pair
// makes the denominator ask for a tau AND its decay products as separate objects, out of
// the same hits, so the efficiency stops meaning anything. A kinematic cut alone does not
// give an antichain.
//
// HardProcess is the OUTGOING LEGS of the hard scatter, not the resonance: the
// deepest-element rule keeps b, b~ and the W decay products on ttbar rather than the
// tops. The resonance itself is the SIGNAL selection, seeded on its PDG ids. Each
// level answers a different question about the same event; none is more correct.

#ifndef PhysicsTools_TruthInfo_interface_TruthLevels_h
#define PhysicsTools_TruthInfo_interface_TruthLevels_h

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "FWCore/Utilities/interface/Exception.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"
#include "SimDataFormats/TruthInfo/interface/Vertex.h"

namespace truth {

  enum class Level {
    StableLegsFromUpstream,
    HardProcess,
    StableDecayProducts,
    CaloBoundary,
    ReconstructableFromSignal,
    UnderlyingEvent,
    PartonJets,
    BHadrons,
    CHadrons,
    ReconstructableFinalState,
    VisibleTau
  };

  // One row per level: the enum value, the bit it stamps on the graph, and the name a
  // configuration selects it by. THE place a level is declared: the name and flag
  // lookups, kAllLevels and the mask fillLevelFlags clears are all derived from it, so
  // adding a level is one row instead of five edits that have to agree. LevelFlag::Signal
  // is deliberately not a row: the selection post-processing owns that bit, not the
  // level machinery.
  struct LevelRow {
    Level level;
    LevelFlag flag;
    char const* name;
  };

  inline constexpr std::array<LevelRow, 11> kLevelTable = {
      {{Level::StableLegsFromUpstream, LevelFlag::StableLegsFromUpstream, "stableLegsFromUpstream"},
       {Level::HardProcess, LevelFlag::HardProcess, "hardProcess"},
       {Level::StableDecayProducts, LevelFlag::StableDecayProducts, "stableDecayProducts"},
       {Level::CaloBoundary, LevelFlag::CaloBoundary, "caloBoundary"},
       {Level::ReconstructableFromSignal, LevelFlag::ReconstructableFromSignal, "reconstructableFromSignal"},
       {Level::UnderlyingEvent, LevelFlag::UnderlyingEvent, "underlyingEvent"},
       {Level::PartonJets, LevelFlag::PartonJets, "partonJets"},
       {Level::BHadrons, LevelFlag::BHadrons, "bHadrons"},
       {Level::CHadrons, LevelFlag::CHadrons, "cHadrons"},
       {Level::ReconstructableFinalState, LevelFlag::ReconstructableFinalState, "reconstructableFinalState"},
       {Level::VisibleTau, LevelFlag::VisibleTau, "visibleTau"}}};

  inline constexpr std::array<Level, kLevelTable.size()> kAllLevels = [] {
    std::array<Level, kLevelTable.size()> levels{};
    for (std::size_t i = 0; i < kLevelTable.size(); ++i) {
      levels[i] = kLevelTable[i].level;
    }
    return levels;
  }();

  // Every bit the level machinery stamps, so fillLevelFlags clears exactly what it owns
  // and a new row cannot leave a stale bit behind.
  inline constexpr uint32_t kOwnedLevelFlags = [] {
    uint32_t mask = 0;
    for (auto const& row : kLevelTable) {
      mask |= static_cast<uint32_t>(row.flag);
    }
    return mask;
  }();

  [[nodiscard]] inline Level levelFromName(std::string const& name) {
    for (auto const& row : kLevelTable) {
      if (name == row.name) {
        return row.level;
      }
    }
    cms::Exception ex("TruthLevels");
    ex << "unknown truth level '" << name << "', expected one of:";
    for (auto const& row : kLevelTable) {
      ex << " " << row.name;
    }
    throw ex;
  }

  // Inverse of levelFromName, so a log line and a configuration string use one spelling.
  [[nodiscard]] inline const char* levelName(Level level) {
    for (auto const& row : kLevelTable) {
      if (row.level == level) {
        return row.name;
      }
    }
    return "unknown";
  }

  namespace detail {
    // reco::GenStatusFlags bit positions, as packed into ParticleData::statusFlags.
    constexpr uint16_t kIsHardProcess = 1u << 7;
    constexpr uint16_t kIsLastCopy = 1u << 13;
  }  // namespace detail

  // Quarks and gluons. Strings, clusters and diquarks are collapsed away by
  // truth::collapseGenShower before the graph is built, so they cannot appear here.
  [[nodiscard]] inline bool isParton(int32_t pdgId) {
    const int64_t a = std::abs(static_cast<int64_t>(pdgId));
    return (a >= 1 && a <= 6) || a == 21;
  }

  // Ordinary hadron whose quark content includes `flavor` (5 = b, 4 = c), read off the
  // PDG hadron-numbering digits. Nuclei and generator-internal codes are not hadrons here.
  [[nodiscard]] inline bool hadronHasQuark(int32_t pdgId, int32_t flavor) {
    const int64_t id = std::abs(static_cast<int64_t>(pdgId));
    if (id < 100 || id >= 1000000000)
      return false;
    // A diquark is nq1 nq2 0 nJ, so its third quark digit is zero. It carries the
    // flavour digits of the hadron it fragments into and would otherwise be flagged
    // AND, being that hadron's ancestor, cover it in the earliest-element antichain.
    if (id >= 1000 && id <= 9999 && (id / 10) % 10 == 0 && (id / 100) % 10 != 0)
      return false;
    const int64_t nq1 = (id / 1000) % 10;
    const int64_t nq2 = (id / 100) % 10;
    const int64_t nq3 = (id / 10) % 10;
    return nq1 == flavor || nq2 == flavor || nq3 == flavor;
  }

  // Whether a seed pdgId list names a RESONANCE to look for.
  //
  // Two spellings mean "no selection" and both must be read that way: an EMPTY list, which
  // is what a production with no preset configures, and {0}, the full-graph escape hatch,
  // since no real particle carries pdgId 0. Neither may be read as "the resonance is
  // missing", and neither may be read as "everything is signal": on such a sample the
  // signal level is NOT ANSWERABLE, so it is not offered at all.
  //
  // Templated because the seed list is std::vector<int> in the module parameters and
  // std::vector<int32_t> on the Graph. One definition, so every consumer decides
  // whether a sample has a resonance by the same rule.
  template <typename Seeds>
  [[nodiscard]] inline bool seedsNameAResonance(Seeds const& seeds) {
    return !seeds.empty() && std::find(seeds.begin(), seeds.end(), 0) == seeds.end();
  }

  // A selection also names a signal when it seeds on heavy-flavour hadron content
  // rather than on pdg ids: the heavyflavor preset carries an empty pdg id list and a
  // flavour list instead, and both spellings must be read the same way everywhere.
  template <typename Seeds, typename Flavors>
  [[nodiscard]] inline bool seedsNameAResonance(Seeds const& seeds, Flavors const& flavors) {
    return seedsNameAResonance(seeds) || !flavors.empty();
  }

  // One entry per physical hadronically decaying tau: the LAST tau of each radiative
  // chain, so a tau radiating a photon counts once, the same last-copy rule the b and c
  // hadron levels use. Requires a GEN decay record, because a tau with no recorded decay
  // cannot be classified, and rejects a decay with an electron or a muon among the
  // children, which is what tau identification measures efficiency against
  // (TauGenJetProducer applies the same rule). Membership alone is an antichain: a tau
  // with a tau child is not a member, so no member can be an ancestor of another member
  // through the only chain taus form.
  [[nodiscard]] inline bool isVisibleTau(Graph const& graph, uint32_t id) {
    auto const& data = graph.particles()[id];
    if (std::abs(static_cast<int64_t>(data.pdgId)) != 15 || data.isSynthetic()) {
      return false;
    }
    bool hasGenDecay = false;
    for (const uint32_t vertexId : graph.decayVertices(id)) {
      if (vertexId >= graph.nVertices() || !graph.vertices()[vertexId].hasGen()) {
        continue;
      }
      hasGenDecay = true;
      for (const uint32_t child : graph.outgoingParticles(vertexId)) {
        if (child >= graph.nParticles() || child == id) {
          continue;
        }
        const int64_t a = std::abs(static_cast<int64_t>(graph.particles()[child].pdgId));
        if (a == 15 || a == 11 || a == 13) {
          return false;
        }
      }
    }
    return hasGenDecay;
  }

  // Whether one particle belongs to a level, before the antichain check.
  [[nodiscard]] inline bool atLevel(Graph const& graph, uint32_t id, Level level) {
    auto const& data = graph.particles()[id];
    switch (level) {
      case Level::StableLegsFromUpstream:
        // Not a per-particle predicate: it is reachability from the Upstream node, so
        // it is answered by stableLegsFromUpstream and never reaches here.
        return false;
      case Level::HardProcess:
        // The hard-scatter legs, not the resonance: see the header note.
        // isHardProcess alone: isHardProcess and isLastCopy are never set on the same
        // copy (0.00 per event on the generator record of ttbar, DYToLL and VBFHZZ4Nu),
        // so repeated copies are removed by the deepest-element antichain below instead.
        return (data.statusFlags & detail::kIsHardProcess) != 0;
      case Level::StableDecayProducts:
        // Final-state generator particles. Stable at GEN means no GEN descendant, so
        // these cannot contain one another.
        return data.hasGen() && data.status == 1;
      case Level::UnderlyingEvent:
        // Reachability from the artificial UnderlyingEvent vertex, answered by
        // stableLegsFromUnderlyingEvent.
        return false;
      case Level::ReconstructableFromSignal:
        // Not a per-particle predicate either: it is a walk down from the signal roots,
        // so it is answered by reconstructableFromSignal and never reaches here.
        return false;
      case Level::PartonJets:
        // Derived from the HardProcess antichain, so it needs that level's result rather
        // than a per-particle rule, and is answered by partonJets().
        return false;
      case Level::BHadrons:
        // The earliest-element antichain then keeps the B* and drops the B below it.
        return hadronHasQuark(data.pdgId, 5);
      case Level::ReconstructableFinalState:
        // A walk from the GEN roots, answered by reconstructableFinalState, so it never
        // reaches here.
        return false;
      case Level::VisibleTau:
        return isVisibleTau(graph, id);
      case Level::CHadrons:
        // A c hadron from a B decay is a legitimate member: the nesting that matters is
        // within one flavour, and beauty and charm are deliberately different levels.
        return hadronHasQuark(data.pdgId, 4);
      case Level::CaloBoundary:
        // Recorded crossing the tracker-calorimeter boundary outward. Back-scattered
        // tracks crossed it inward and are the same particle coming back.
        return !data.backscattered && Particle(&graph, id).checkpoint(0).has_value();
    }
    return false;
  }

  // Stable legs hanging off every artificial vertex of one role. Upstream collects the
  // ISR and upstream side of the interaction, UnderlyingEvent the spectators; the walk is
  // identical, so it is written once. A leg is a particle that produced nothing further,
  // which makes the result an antichain by construction.
  [[nodiscard]] inline std::vector<uint32_t> stableLegsFromRole(Graph const& graph, VertexRole role) {
    std::vector<uint32_t> legs;
    std::vector<bool> seen(graph.nParticles(), false);
    std::vector<uint32_t> stack;

    const uint32_t nVertices = graph.nVertices();
    for (uint32_t v = 0; v < nVertices; ++v) {
      auto const& vertexData = graph.vertices()[v];
      if (vertexData.vertexRole() != role) {
        continue;
      }
      // Depth-first from each outgoing particle over the raw CSR spans; a particle
      // the GENERATOR gave nothing further is a leg. Only GEN decay vertices count
      // and are descended: a SIM continuation is transport, so a stable ISR photon
      // that converts in the tracker stays the leg instead of dissolving into its
      // conversion products.
      for (const uint32_t outgoing : graph.outgoingParticles(v)) {
        stack.push_back(outgoing);
      }
      while (!stack.empty()) {
        const uint32_t id = stack.back();
        stack.pop_back();
        if (id >= seen.size() || seen[id]) {
          continue;
        }
        seen[id] = true;
        bool isLeg = true;
        for (const uint32_t vertexId : graph.decayVertices(id)) {
          if (vertexId >= nVertices || !graph.vertices()[vertexId].hasGen()) {
            continue;
          }
          for (const uint32_t child : graph.outgoingParticles(vertexId)) {
            if (child == id) {
              continue;
            }
            isLeg = false;
            if (child < seen.size() && !seen[child]) {
              stack.push_back(child);
            }
          }
        }
        if (isLeg) {
          legs.push_back(id);
        }
      }
    }
    std::sort(legs.begin(), legs.end());
    return legs;
  }

  [[nodiscard]] inline std::vector<uint32_t> stableLegsFromUpstream(Graph const& graph) {
    return stableLegsFromRole(graph, VertexRole::Upstream);
  }

  [[nodiscard]] inline std::vector<uint32_t> stableLegsFromUnderlyingEvent(Graph const& graph) {
    return stableLegsFromRole(graph, VertexRole::UnderlyingEvent);
  }

  // Species a detector cannot reconstruct at all, so they are not part of the visible
  // final state. Only the neutrinos today; anything else invisible would belong here.
  [[nodiscard]] inline bool isInvisible(int32_t pdgId) {
    const int64_t a = std::abs(static_cast<int64_t>(pdgId));
    return a == 12 || a == 14 || a == 16;
  }

  namespace detail {
    // The reconstructable-final-state walk from a caller-chosen seed set. The seed
    // predicate is the ONLY difference between the signal-seeded level and the
    // event-wide one, so the walk and its termination rules live here once.
    template <typename SeedPredicate>
    [[nodiscard]] inline std::vector<uint32_t> reconstructableLegsFrom(Graph const& graph, SeedPredicate isSeed) {
      const uint32_t nParticles = graph.nParticles();
      std::vector<uint32_t> legs;
      std::vector<bool> seen(nParticles, false);
      std::vector<uint32_t> stack;
      auto const& terminating = graph.reconstructablePdgIds();

      for (uint32_t p = 0; p < nParticles; ++p) {
        if (isSeed(p)) {
          seen[p] = true;
          stack.push_back(p);
        }
      }

      while (!stack.empty()) {
        const uint32_t p = stack.back();
        stack.pop_back();
        auto const& data = graph.particles()[p];

        // Terminal three ways: the detector reconstructs this species as an object even
        // though it decays (pi0), the generator called it stable, or the generator wrote
        // nothing below it. Anything else is an intermediate the detector never sees as
        // an object, an a1 or a rho, and the walk goes through it without labelling it.
        // The walk descends through GEN decay vertices ONLY: this level is the visible
        // final state of the GENERATOR, and a SIM continuation is transport, not decay.
        // A K0S the generator decayed but Geant4 also interacted in material must yield
        // its GEN pions, never the nuclear secondaries of the SIM vertex.
        // The seen mask makes this terminate on a graph with a cycle.
        const bool reconstructableSpecies =
            std::find(terminating.begin(), terminating.end(), data.pdgId) != terminating.end();
        const bool genStable = data.hasGen() && data.status == 1;
        bool hasGenDecay = false;
        for (const uint32_t vertexId : graph.decayVertices(p)) {
          if (vertexId < graph.nVertices() && graph.vertices()[vertexId].hasGen()) {
            hasGenDecay = true;
            break;
          }
        }
        if (reconstructableSpecies || genStable || !hasGenDecay) {
          // A synthetic particle is an accounting object with no hits, so it can never be
          // reconstructed and must not become a leg: the signal stand-in is Signal-flagged
          // and vertex-less, and both terminals above are true for it.
          if (!isInvisible(data.pdgId) && !data.isSynthetic()) {
            legs.push_back(p);
          }
          continue;
        }

        for (const uint32_t vertexId : graph.decayVertices(p)) {
          if (vertexId >= graph.nVertices() || !graph.vertices()[vertexId].hasGen()) {
            continue;
          }
          for (const uint32_t child : graph.outgoingParticles(vertexId)) {
            if (child < nParticles && !seen[child]) {
              seen[child] = true;
              stack.push_back(child);
            }
          }
        }
      }

      std::sort(legs.begin(), legs.end());
      return legs;
    }
  }  // namespace detail

  // The first stable, reconstructable particles the signal produced.
  //
  // Walk down from every Signal root and stop at the first generator-stable descendant,
  // which is where the decay chain ends and the detector's job begins. GEN-stable
  // terminates the walk on purpose: a stable pion still has a SIM continuation as it
  // showers, and descending into that would return shower fragments instead of the
  // particle the resonance actually produced.
  //
  // Neutrinos are dropped rather than walked through, so the result is the VISIBLE final
  // state of the resonance. A signal root that is itself stable, a gun electron say, is
  // its own leg.
  //
  // An antichain by construction: the walk stops at each leg, so no leg can be an
  // ancestor of another. Empty when nothing carries the Signal flag.
  [[nodiscard]] inline std::vector<uint32_t> reconstructableFromSignal(Graph const& graph) {
    return detail::reconstructableLegsFrom(
        graph, [&graph](uint32_t p) { return graph.particles()[p].isAtLevel(LevelFlag::Signal); });
  }

  // The same walk seeded from every GEN root, the particles with a GEN record and no GEN
  // parent, so the level exists on every sample: a pi0 is one object inside a QCD jet,
  // the underlying event and each pileup interaction, none of which has a resonance to
  // seed from. reconstructableFromSignal answers "what did the resonance produce"; this
  // level answers "what could the detector see", event-wide.
  [[nodiscard]] inline std::vector<uint32_t> reconstructableFinalState(Graph const& graph) {
    auto const isGenRoot = [&graph](uint32_t p) {
      if (!graph.particles()[p].hasGen()) {
        return false;
      }
      for (const uint32_t vertexId : graph.productionVertices(p)) {
        if (vertexId >= graph.nVertices()) {
          continue;
        }
        for (const uint32_t parent : graph.incomingParticles(vertexId)) {
          if (parent != p && graph.particles()[parent].hasGen()) {
            return false;
          }
        }
      }
      return true;
    };
    return detail::reconstructableLegsFrom(graph, isGenRoot);
  }

  // PartonJets is defined in terms of the HardProcess antichain and levelAntichain
  // dispatches back to it, so one of the two has to be declared ahead of the other.
  [[nodiscard]] inline std::vector<uint32_t> levelAntichain(Graph const& graph, Level level);

  // One root per parton-initiated jet: the hard-scatter legs that are partons, each
  // standing for its descendant subgraph. No clustering and no cone; the flavour is the
  // parton's own PDG id. The deepest-element rule of HardProcess keeps a top's b rather
  // than the top and keeps the incoming beam partons out. EMPTY, not wrong, when
  // statusFlags are unavailable, which is the HepMC3 path. The flag-driven levels are
  // NOT restricted to the signal interaction: they hold whatever carries isHardProcess,
  // and a consumer that needs a signal-only set filters on eventId as the denominator
  // producer does. Measured on 10 PU200 ttbar events with the standard pile-up library,
  // no overlaid interaction carries the flag, so the level is signal-only in practice.
  // The ROOTS are an
  // antichain but the SUBGRAPHS may overlap: two colour-connected quarks fragment
  // through one string, and assigning each hadron to exactly one jet is what a
  // clustering algorithm is for.
  [[nodiscard]] inline std::vector<uint32_t> partonJets(Graph const& graph) {
    std::vector<uint32_t> roots = levelAntichain(graph, Level::HardProcess);
    roots.erase(
        std::remove_if(
            roots.begin(), roots.end(), [&graph](uint32_t id) { return !isParton(graph.particles()[id].pdgId); }),
        roots.end());
    return roots;
  }

  // The level as an antichain. Candidates that have another candidate as an ancestor are
  // dropped, so what remains is one entry per physical object at that level. The
  // membership rules above are already antichains in a well-formed graph; the check is
  // kept because a denominator that silently contains a particle and its own parent is
  // the failure this class exists to prevent.
  // Drop every member that another member covers. With keepDeepest false a member that
  // has a member ANCESTOR goes, which leaves the earliest of each chain; keepDeepest
  // reverses the direction. THIS is what makes a level an antichain, so every level runs
  // it. A membership rule that looks like an antichain is not enough: on a re-convergent
  // history a walk that stops at a pi0 on one path still reaches that pi0's photon on
  // another, and the level ends up holding both.
  inline void dropCoveredMembers(Graph const& graph, std::vector<uint32_t>& members, bool keepDeepest) {
    const uint32_t nParticles = graph.nParticles();
    std::vector<uint8_t> covered(nParticles, 0);
    std::vector<uint32_t> stack;
    stack.reserve(members.size());
    // Seed with the members' immediate neighbours in the chosen direction, so a
    // member itself is only marked when REACHED from another member.
    auto pushNeighbours = [&](uint32_t id) {
      if (keepDeepest) {
        for (const uint32_t vertexId : graph.productionVertices(id)) {
          if (vertexId >= graph.nVertices()) {
            continue;
          }
          for (const uint32_t parent : graph.incomingParticles(vertexId)) {
            // A particle that is its own neighbour would cover itself and drop out of
            // its own level; the graph navigation guards self-loops the same way.
            if (parent != id && parent < nParticles && covered[parent] == 0) {
              covered[parent] = 1;
              stack.push_back(parent);
            }
          }
        }
      } else {
        for (const uint32_t vertexId : graph.decayVertices(id)) {
          if (vertexId >= graph.nVertices()) {
            continue;
          }
          for (const uint32_t child : graph.outgoingParticles(vertexId)) {
            if (child != id && child < nParticles && covered[child] == 0) {
              covered[child] = 1;
              stack.push_back(child);
            }
          }
        }
      }
    };
    for (uint32_t id : members) {
      pushNeighbours(id);
    }
    while (!stack.empty()) {
      const uint32_t id = stack.back();
      stack.pop_back();
      pushNeighbours(id);
    }

    std::erase_if(members, [&covered](uint32_t id) { return covered[id] != 0; });
  }

  [[nodiscard]] inline std::vector<uint32_t> levelAntichain(Graph const& graph, Level level) {
    if (level == Level::StableLegsFromUpstream) {
      std::vector<uint32_t> legs = stableLegsFromUpstream(graph);
      dropCoveredMembers(graph, legs, false);
      return legs;
    }
    if (level == Level::ReconstructableFromSignal) {
      std::vector<uint32_t> legs = reconstructableFromSignal(graph);
      dropCoveredMembers(graph, legs, false);
      return legs;
    }
    if (level == Level::ReconstructableFinalState) {
      std::vector<uint32_t> legs = reconstructableFinalState(graph);
      dropCoveredMembers(graph, legs, false);
      return legs;
    }
    if (level == Level::UnderlyingEvent) {
      std::vector<uint32_t> legs = stableLegsFromUnderlyingEvent(graph);
      dropCoveredMembers(graph, legs, false);
      return legs;
    }
    if (level == Level::PartonJets) {
      // Filtered from HardProcess, which dropCoveredMembers already reduced.
      return partonJets(graph);
    }
    std::vector<uint32_t> candidates;
    const uint32_t nParticles = graph.nParticles();
    for (uint32_t id = 0; id < nParticles; ++id) {
      if (atLevel(graph, id, level)) {
        candidates.push_back(id);
      }
    }
    // Which end of a chain of candidates to keep, per level.
    //
    // Earliest, the default: the members are final states, and a candidate with a
    // candidate ANCESTOR is a duplicate of it.
    //
    // Deepest for HardProcess: the incoming partons and the outgoing particles both carry
    // the flag, the incoming ones are ancestors of the outgoing ones, and it is the
    // outgoing ones that the level is about. Keeping the earliest there would return the
    // beam partons, which sit at pt 0 and enormous eta and are then dropped by any
    // kinematic selector, leaving the level empty.
    //
    // Deepest for BHadrons and CHadrons: the object is the hadron that DECAYS WEAKLY, so
    // the level names the same particle CMS names. A B* radiating to a B is not a
    // duplicate of it: the two carry different momenta and, decisively, different decay
    // vertices, because the B* decays electromagnetically at the production point while
    // the B travels. Measured on 200 ttbar and 300 QCD generator events: the count is the
    // same either way, 68.9% and 61.8% of chains hold a different particle, and the
    // median decay displacement goes from 0.000 cm to 0.46 cm.
    const bool keepDeepest = level == Level::HardProcess || level == Level::BHadrons || level == Level::CHadrons;
    dropCoveredMembers(graph, candidates, keepDeepest);
    return candidates;
  }

  // The persisted bit for a level. Kept next to the Level enum so adding a level forces
  // the author past this switch, which has no default for that reason.
  [[nodiscard]] inline LevelFlag levelFlagOf(Level level) {
    for (auto const& row : kLevelTable) {
      if (row.level == level) {
        return row.flag;
      }
    }
    throw cms::Exception("TruthLevels") << "level " << static_cast<int>(level) << " has no row in kLevelTable";
  }

  // Stamp every particle with the levels it belongs to. Call once, on the COMPLETE graph:
  // levelAntichain walks ancestors and descendants, so a graph still being assembled
  // gives an antichain of whatever existed at the time.
  //
  // Clears first, so calling it twice is the same as calling it once. That matters
  // because a stale flag is indistinguishable from a fresh one by inspection, and the
  // only defence is that the operation is reproducible and idempotent.
  inline void fillLevelFlags(Graph& graph) {
    // Preconditions, because the walks below index the CSR arrays directly: the graph
    // must be shaped (Graph::isConsistent) and acyclic. A short offset array aborts the
    // job with a bare std::out_of_range and no module context. A cycle makes every member
    // of a level cover every other, which empties that level with no diagnostic.
    // TruthGraphTopologyChecker counts cycles in a job.
    if (graph.nParticles() == 0) {
      return;
    }
    if (graph.particleToDecayVertexOffsets().size() != static_cast<std::size_t>(graph.nParticles()) + 1 ||
        graph.particleToProductionVertexOffsets().size() != static_cast<std::size_t>(graph.nParticles()) + 1) {
      throw cms::Exception("TruthLevels")
          << "fillLevelFlags needs CSR offsets of size nParticles + 1 (" << graph.nParticles() + 1 << "), found "
          << graph.particleToDecayVertexOffsets().size() << " and " << graph.particleToProductionVertexOffsets().size();
    }
    // Clear only the bits this function owns. LevelFlag::Signal is set upstream, by the
    // selection post-processing that knows the seed species, and clearing it here would
    // silently erase the resonance.
    for (auto& particle : graph.particles()) {
      particle.levelFlags &= ~kOwnedLevelFlags;
    }
    // The HardProcess antichain feeds two levels, itself and (filtered to partons)
    // the parton jets, so it is computed once.
    const std::vector<uint32_t> hardProcess = levelAntichain(graph, Level::HardProcess);
    for (const Level level : kAllLevels) {
      const LevelFlag flag = levelFlagOf(level);
      std::vector<uint32_t> ids;
      if (level == Level::HardProcess) {
        ids = hardProcess;
      } else if (level == Level::PartonJets) {
        ids = hardProcess;
        ids.erase(std::remove_if(
                      ids.begin(), ids.end(), [&graph](uint32_t id) { return !isParton(graph.particles()[id].pdgId); }),
                  ids.end());
      } else {
        ids = levelAntichain(graph, level);
      }
      for (const uint32_t id : ids) {
        if (id < graph.nParticles()) {
          graph.particles()[id].setLevel(flag);
        }
      }
    }
  }

}  // namespace truth

#endif
