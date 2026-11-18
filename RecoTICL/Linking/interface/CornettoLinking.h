#ifndef RecoTICL_Linking_CornettoLinking_h
#define RecoTICL_Linking_CornettoLinking_h

// Pair test, parent search and emission rule of the Cornetto linking.

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "DataFormats/HGCalReco/interface/Common.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"
#include "RecoTICL/Linking/interface/CornettoFeatures.h"
#include "RecoTICL/Linking/interface/CornettoGeometry.h"
#include "RecoTICL/Linking/interface/CornettoGrouping.h"

namespace ticl::cornetto {

  // Cuts of the pair test that do not depend on the cone.
  struct PairCuts {
    float etaWindow;            // max |deta| and |dphi| between the barycenters
    float typeVetoProbability;  // 0 disables the type veto and the EM cone
    bool useTimeGate;
    float timeNSigma2;  // square of the max time difference in combined sigmas
    float timeFloor2;   // square of the time resolution [ns] added to the combined error
  };

  // True when t_other - t_anchor - path / c agrees with 0 within the time gate, or when the gate is off or an
  // error is not positive.
  bool timeCompatible(const PairCuts& cuts, float tAnchor, float errAnchor, float tOther, float errOther, float path);

  // True when b passes the pair test with the anchor a: same endcap, |deta| and |dphi| at most etaWindow, b inside
  // the cone of a (emCone for an EM anchor), the time gate and the type veto. A NaN position fails the pair test.
  bool compatible(
      const PairCuts& cuts, const Cone& cone, const Cone& emCone, const Features& f, unsigned int a, unsigned int b);

  struct EmissionRule {
    float minEnergy;
    float minPt;
    float minEnergyEM;
    float minPtEM;
    float minEnergyAnyPt;  // values <= 0 disable it
    float typeVetoProbability;
  };

  // True when the raw energy is at least minEnergy, and the raw pt is at least minPt or the raw energy is at least a
  // positive minEnergyAnyPt. A group with em above a positive typeVetoProbability uses minEnergyEM and minPtEM.
  inline bool emits(const EmissionRule& rule, float energy, float pt, float em) {
    const bool isEM = rule.typeVetoProbability > 0.f && em > rule.typeVetoProbability;
    if (!(energy >= (isEM ? rule.minEnergyEM : rule.minEnergy)))
      return false;
    return pt >= (isEM ? rule.minPtEM : rule.minPt) || (rule.minEnergyAnyPt > 0.f && energy >= rule.minEnergyAnyPt);
  }

  using EndcapTiles = std::array<TICLLayerTile, 2>;  // one per endcap, index 1 for z > 0

  // Fills each valid node into the tile of its endcap in decreasing rank, so each bin lists its nodes in that order.
  void fillTiles(const Features& f, EndcapTiles& tiles);

  // The parent of each valid node: the highest-ranked node a above it with allowed(a, node) and test(a, node) true.
  // Tiles from fillTiles; test must refuse pairs across endcaps, outside etaWindow or outside cone.
  template <typename Allowed, typename Test>
  std::vector<unsigned int> findParents(const Features& f,
                                        const EndcapTiles& tiles,
                                        const Cone& cone,
                                        float etaWindow,
                                        const Allowed& allowed,
                                        const Test& test) {
    const auto n = static_cast<unsigned int>(f.energy.size());
    const auto above = [&f](unsigned int a, unsigned int b) { return ranksAbove(f.energy, a, b); };
    std::vector<unsigned int> parents(n);
    for (unsigned int i = 0; i < n; ++i) {
      unsigned int best = i;
      if (f.valid[i] && f.rad[i] > 0.f) {
        const float absEta = std::abs(f.eta[i]);
        const auto window = searchWindow(cone, absEta, f.rad[i], etaWindow);
        const float etaMin = std::clamp(absEta - window.dEta, TileConstants::minEta, TileConstants::maxEta);
        const float etaMax = std::clamp(absEta + window.dEta, TileConstants::minEta, TileConstants::maxEta);
        auto const& tile = tiles[f.bary[i].z() > 0.f ? 1 : 0];
        const auto box = tile.searchBoxEtaPhi(etaMin, etaMax, f.phi[i] - window.dPhi, f.phi[i] + window.dPhi);
        for (int etaBin = box[0]; etaBin <= box[1]; ++etaBin) {
          for (int phiBin = box[2]; phiBin <= box[3]; ++phiBin) {
            for (unsigned int j : tile[tile.globalBin(etaBin, phiBin % TileConstants::nPhiBins)]) {
              // The bin lists its nodes in decreasing rank, so no later node in the bin can be a better parent.
              if (!above(j, i) || (best != i && !above(j, best)))
                break;
              if (allowed(j, i) && test(j, i))
                best = j;
            }
          }
        }
      }
      parents[i] = best;
    }
    return parents;
  }

}  // namespace ticl::cornetto

#endif
