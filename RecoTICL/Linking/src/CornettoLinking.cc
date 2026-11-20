#include <algorithm>
#include <cmath>
#include <vector>

#include "DataFormats/Math/interface/deltaPhi.h"
#include "RecoTICL/Linking/interface/CornettoLinking.h"

namespace ticl::cornetto {

  bool timeCompatible(const PairCuts& cuts, float tAnchor, float errAnchor, float tOther, float errOther, float path) {
    if (!cuts.useTimeGate || !(errAnchor > 0.f) || !(errOther > 0.f))
      return true;
    const float dt = tOther - tAnchor - path * kInvSpeedOfLight;
    return dt * dt <= cuts.timeNSigma2 * (errAnchor * errAnchor + errOther * errOther + cuts.timeFloor2);
  }

  // A NaN time error or EM probability disables the time gate or the type veto for that pair.
  bool compatible(
      const PairCuts& cuts, const Cone& cone, const Cone& emCone, const Features& f, unsigned int a, unsigned int b) {
    if (!(f.bary[a].z() * f.bary[b].z() > 0.f))
      return false;  // same endcap only
    if (!(std::abs(f.eta[b] - f.eta[a]) <= cuts.etaWindow) ||
        !(std::abs(reco::deltaPhi(f.phi[b], f.phi[a])) <= cuts.etaWindow))
      return false;
    const float veto = cuts.typeVetoProbability;
    const bool emAnchor = veto > 0.f && f.em[a] > veto;
    const auto s = separationInCone(emAnchor ? emCone : cone, f.bary[a], f.rad[a], f.bary[b]);
    if (!s)
      return false;
    if (!timeCompatible(cuts, f.time[a], f.timeError[a], f.time[b], f.timeError[b], *s))
      return false;
    if (veto > 0.f && f.em[a] >= 0.f && f.em[b] >= 0.f) {
      const float hi = veto;
      const float lo = 1.f - veto;
      if ((f.em[a] > hi && f.em[b] < lo) || (f.em[a] < lo && f.em[b] > hi))
        return false;
    }
    return true;
  }

  void fillTiles(const Features& f, EndcapTiles& tiles) {
    std::vector<unsigned int> order;
    order.reserve(f.energy.size());
    for (unsigned int i = 0; i < f.energy.size(); ++i)
      if (f.valid[i])
        order.push_back(i);
    std::sort(order.begin(), order.end(), [&f](unsigned int a, unsigned int b) { return ranksAbove(f.energy, a, b); });
    for (unsigned int i : order)
      tiles[f.bary[i].z() > 0.f ? 1 : 0].fill(f.eta[i], f.phi[i], i);
  }

}  // namespace ticl::cornetto
