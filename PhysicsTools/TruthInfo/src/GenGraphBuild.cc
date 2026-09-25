// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <unordered_set>

#include "DataFormats/HepMCCandidate/interface/GenStatusFlags.h"
#include "PhysicsTools/HepMCCandAlgos/interface/MCTruthHelper.h"
#include "PhysicsTools/TruthInfo/interface/GenGraphBuild.h"

#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"
#include "HepMC/GenVertex.h"
#include "HepMC3/GenEvent.h"
#include "HepMC3/GenParticle.h"
#include "HepMC3/GenVertex.h"

namespace {

  constexpr uint16_t kIsHardProcess = 1u << reco::GenStatusFlags::kIsHardProcess;
  constexpr uint16_t kIsLastCopy = 1u << reco::GenStatusFlags::kIsLastCopy;

  // Sentinel for "this particle has no production vertex in the record", which is the
  // case for the incoming beam particles.
  constexpr int kNoVertex = std::numeric_limits<int>::min();

  // Shower bookkeeping objects rather than particles reconstruction can be asked
  // about: partons, diquarks, Pythia strings and clusters, and the beam/system
  // pseudoparticles. Their last copy carries no physics of its own.
  [[nodiscard]] bool isShowerObject(int32_t pdgId) {
    const int32_t id = std::abs(pdgId);
    if (id >= 1 && id <= 6)
      return true;
    if (id == 21)
      return true;
    if (id >= 91 && id <= 94)  // cluster, string and the other hadronization placeholders
      return true;
    if (id == 990)  // pomeron
      return true;
    if (id >= 1000 && id <= 9999 && (id / 10) % 10 == 0 && (id / 100) % 10 != 0)  // diquarks, e.g. 2101, 2203
      return true;
    if (id >= 9900000 && id < 1000000000)  // generator-internal states such as 9922212, below the nuclei codes
      return true;
    return false;
  }

  template <typename V>
  [[nodiscard]] V lookup(std::unordered_map<int, V> const& map, int key, V fallback) {
    const auto it = map.find(key);
    return it != map.end() ? it->second : fallback;
  }

  [[nodiscard]] bool keepGenParticle(truth::GenBuild const& gb,
                                     int barcode,
                                     std::unordered_set<int> const& simContinuedBarcodes) {
    if (simContinuedBarcodes.count(barcode) != 0)
      return true;
    if (lookup(gb.particleStatusByBarcode, barcode, static_cast<int16_t>(0)) == 1)
      return true;
    const uint16_t flags = lookup(gb.particleStatusFlagsByBarcode, barcode, static_cast<uint16_t>(0));
    if ((flags & kIsHardProcess) != 0)
      return true;
    return (flags & kIsLastCopy) != 0 && !isShowerObject(lookup(gb.particlePdgIdByBarcode, barcode, int32_t{0}));
  }

  void sortUnique(std::vector<uint32_t>& v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
  }

  template <typename V>
  void pruneMap(std::unordered_map<int, V>& map, std::unordered_set<int> const& keptBarcodes) {
    for (auto it = map.begin(); it != map.end();)
      it = (keptBarcodes.count(it->first) == 0) ? map.erase(it) : std::next(it);
  }

}  // namespace

namespace truth {

  GenBuild buildFromHepMC2(HepMC::GenEvent const& ev) {
    GenBuild gb;

    std::unordered_set<int> seenV;
    std::unordered_set<int> seenP;

    gb.particlePdgIdByBarcode.reserve(ev.particles_size() * 2);
    gb.particleStatusByBarcode.reserve(ev.particles_size() * 2);
    gb.particleBarcodeByIndex.reserve(ev.particles_size());

    for (auto v = ev.vertices_begin(); v != ev.vertices_end(); ++v) {
      if (*v == nullptr)
        continue;

      const int vbc = (*v)->barcode();

      if (seenV.insert(vbc).second)
        gb.vtxBarcodes.push_back(vbc);

      for (auto po = (*v)->particles_out_const_begin(); po != (*v)->particles_out_const_end(); ++po) {
        if (*po == nullptr)
          continue;

        const int pbc = (*po)->barcode();

        if (seenP.insert(pbc).second)
          gb.partBarcodes.push_back(pbc);

        gb.vtxToPart.emplace_back(vbc, pbc);
      }

      for (auto pi = (*v)->particles_in_const_begin(); pi != (*v)->particles_in_const_end(); ++pi) {
        if (*pi == nullptr)
          continue;

        const int pbc = (*pi)->barcode();

        if (seenP.insert(pbc).second)
          gb.partBarcodes.push_back(pbc);

        gb.partToVtx.emplace_back(pbc, vbc);
      }
    }

    MCTruthHelper<HepMC::GenParticle> mcTruthHelper;
    for (auto p = ev.particles_begin(); p != ev.particles_end(); ++p) {
      if (*p == nullptr)
        continue;

      const int pbc = (*p)->barcode();

      gb.particleBarcodeByIndex.push_back(pbc);
      gb.particlePdgIdByBarcode.emplace(pbc, (*p)->pdg_id());
      gb.particleStatusByBarcode.emplace(pbc, static_cast<int16_t>((*p)->status()));

      reco::GenStatusFlags flags;
      mcTruthHelper.fillGenStatusFlags(**p, flags);
      gb.particleStatusFlagsByBarcode.emplace(pbc, static_cast<uint16_t>(flags.flags_.to_ulong()));

      if (seenP.insert(pbc).second)
        gb.partBarcodes.push_back(pbc);
    }

    return gb;
  }

  GenBuild buildFromHepMC3(HepMC3::GenEvent const& ev) {
    GenBuild gb;

    std::unordered_set<int> seenV;
    std::unordered_set<int> seenP;

    gb.particlePdgIdByBarcode.reserve(ev.particles().size() * 2);
    gb.particleStatusByBarcode.reserve(ev.particles().size() * 2);
    gb.particleBarcodeByIndex.reserve(ev.particles().size());

    for (auto const& vptr : ev.vertices()) {
      if (!vptr)
        continue;

      const int vbc = vptr->id();

      if (seenV.insert(vbc).second)
        gb.vtxBarcodes.push_back(vbc);

      for (auto const& po : vptr->particles_out()) {
        if (!po)
          continue;

        const int pbc = po->id();

        if (seenP.insert(pbc).second)
          gb.partBarcodes.push_back(pbc);

        gb.vtxToPart.emplace_back(vbc, pbc);
      }

      for (auto const& pi : vptr->particles_in()) {
        if (!pi)
          continue;

        const int pbc = pi->id();

        if (seenP.insert(pbc).second)
          gb.partBarcodes.push_back(pbc);

        gb.partToVtx.emplace_back(pbc, vbc);
      }
    }

    for (auto const& pptr : ev.particles()) {
      if (!pptr)
        continue;

      const int pbc = pptr->id();

      gb.particleBarcodeByIndex.push_back(pbc);
      gb.particlePdgIdByBarcode.emplace(pbc, pptr->pid());
      gb.particleStatusByBarcode.emplace(pbc, static_cast<int16_t>(pptr->status()));

      if (seenP.insert(pbc).second)
        gb.partBarcodes.push_back(pbc);
    }

    return gb;
  }

  bool collapseGenShower(GenBuild& gb, std::unordered_set<int> const& simContinuedBarcodes) {
    const uint32_t nPart = static_cast<uint32_t>(gb.partBarcodes.size());
    if (nPart == 0)
      return true;

    const bool statusFlagsAvailable = !gb.particleStatusFlagsByBarcode.empty();

    std::unordered_map<int, uint32_t> indexOfBarcode;
    indexOfBarcode.reserve(nPart * 2);
    for (uint32_t i = 0; i < nPart; ++i)
      indexOfBarcode.emplace(gb.partBarcodes[i], i);

    std::vector<int> prodVertexOf(nPart, kNoVertex);
    for (auto const& [vbc, pbc] : gb.vtxToPart) {
      const auto it = indexOfBarcode.find(pbc);
      if (it != indexOfBarcode.end() && prodVertexOf[it->second] == kNoVertex)
        prodVertexOf[it->second] = vbc;
    }

    std::unordered_map<int, std::vector<uint32_t>> incomingOf;
    incomingOf.reserve(gb.vtxBarcodes.size() * 2);
    for (auto const& [pbc, vbc] : gb.partToVtx) {
      const auto it = indexOfBarcode.find(pbc);
      if (it != indexOfBarcode.end())
        incomingOf[vbc].push_back(it->second);
    }

    const std::vector<uint32_t> noParents;
    auto parentsOf = [&](uint32_t i) -> std::vector<uint32_t> const& {
      if (prodVertexOf[i] == kNoVertex)
        return noParents;
      const auto it = incomingOf.find(prodVertexOf[i]);
      return it != incomingOf.end() ? it->second : noParents;
    };

    std::vector<uint8_t> keep(nPart, 0);
    for (uint32_t i = 0; i < nPart; ++i)
      keep[i] = keepGenParticle(gb, gb.partBarcodes[i], simContinuedBarcodes) ? 1 : 0;

    // Nearest surviving ancestors of every particle, over the parent relation (the
    // particles incoming to its production vertex). Iterative and memoized; a node
    // still being computed is skipped, so a malformed record with a cycle terminates.
    std::vector<std::vector<uint32_t>> nearestKept(nPart);
    std::vector<uint8_t> state(nPart, 0);  // 0 = new, 1 = in progress, 2 = done
    std::vector<uint32_t> stack;

    for (uint32_t seed = 0; seed < nPart; ++seed) {
      if (state[seed] == 2)
        continue;
      stack.push_back(seed);

      while (!stack.empty()) {
        const uint32_t i = stack.back();

        if (state[i] == 2) {
          stack.pop_back();
          continue;
        }

        if (state[i] == 0) {
          state[i] = 1;
          if (keep[i] != 0) {
            nearestKept[i].push_back(i);
            state[i] = 2;
            stack.pop_back();
            continue;
          }
          for (const uint32_t parent : parentsOf(i)) {
            if (state[parent] == 0)
              stack.push_back(parent);
          }
          continue;
        }

        for (const uint32_t parent : parentsOf(i)) {
          if (state[parent] == 2)
            nearestKept[i].insert(nearestKept[i].end(), nearestKept[parent].begin(), nearestKept[parent].end());
        }
        sortUnique(nearestKept[i]);
        state[i] = 2;
        stack.pop_back();
      }
    }

    // A vertex survives when it still produces a survivor. A survivor with no
    // production vertex keeps the record's own topology: it had no incoming edge
    // before the collapse either.
    std::unordered_set<int> keptVertices;
    std::vector<std::pair<int, int>> vtxToPart;
    for (uint32_t i = 0; i < nPart; ++i) {
      if (keep[i] == 0 || prodVertexOf[i] == kNoVertex)
        continue;
      keptVertices.insert(prodVertexOf[i]);
      vtxToPart.emplace_back(prodVertexOf[i], gb.partBarcodes[i]);
    }

    // The surviving vertex inherits the nearest surviving ancestors of everything that
    // fed it, which is the contraction: the collapsed chain becomes one edge. A vertex
    // left without any is a root, and the callers attach the GenEvent node to it.
    std::vector<std::pair<int, int>> partToVtx;
    std::vector<uint32_t> ancestors;
    for (const int vbc : gb.vtxBarcodes) {
      if (keptVertices.count(vbc) == 0)
        continue;
      const auto it = incomingOf.find(vbc);
      if (it == incomingOf.end())
        continue;
      ancestors.clear();
      for (const uint32_t parent : it->second)
        ancestors.insert(ancestors.end(), nearestKept[parent].begin(), nearestKept[parent].end());
      sortUnique(ancestors);
      for (const uint32_t a : ancestors) {
        if (prodVertexOf[a] != vbc)  // a vertex must not become its own ancestor
          partToVtx.emplace_back(gb.partBarcodes[a], vbc);
      }
    }

    std::unordered_set<int> keptBarcodes;
    keptBarcodes.reserve(nPart * 2);
    std::vector<int> partBarcodes;
    for (uint32_t i = 0; i < nPart; ++i) {
      if (keep[i] == 0)
        continue;
      partBarcodes.push_back(gb.partBarcodes[i]);
      keptBarcodes.insert(gb.partBarcodes[i]);
    }

    std::vector<int> vtxBarcodes;
    for (const int vbc : gb.vtxBarcodes) {
      if (keptVertices.count(vbc) != 0)
        vtxBarcodes.push_back(vbc);
    }

    std::vector<int> particleBarcodeByIndex;
    for (const int pbc : gb.particleBarcodeByIndex) {
      if (keptBarcodes.count(pbc) != 0)
        particleBarcodeByIndex.push_back(pbc);
    }

    pruneMap(gb.particlePdgIdByBarcode, keptBarcodes);
    pruneMap(gb.particleStatusByBarcode, keptBarcodes);
    pruneMap(gb.particleStatusFlagsByBarcode, keptBarcodes);

    gb.partBarcodes = std::move(partBarcodes);
    gb.vtxBarcodes = std::move(vtxBarcodes);
    gb.particleBarcodeByIndex = std::move(particleBarcodeByIndex);
    gb.vtxToPart = std::move(vtxToPart);
    gb.partToVtx = std::move(partToVtx);

    return statusFlagsAvailable;
  }

}  // namespace truth
