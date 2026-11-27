#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HGCalReco/interface/Common.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "TracksterLinkingbyCornetto.h"

using namespace ticl;

TracksterLinkingbyCornetto::TracksterLinkingbyCornetto(const edm::ParameterSet& conf,
                                                       edm::ConsumesCollector iC,
                                                       cms::Ort::ONNXRuntime const* onnxRuntime)
    : TracksterLinkingAlgoBase(conf, iC, onnxRuntime),
      cuts_{conf.getParameter<float>("etaWindow"),
            conf.getParameter<float>("typeVetoProbability"),
            conf.getParameter<float>("timeCompatibilityNSigma") > 0.f,
            conf.getParameter<float>("timeCompatibilityNSigma") * conf.getParameter<float>("timeCompatibilityNSigma"),
            conf.getParameter<float>("timeResolutionFloor") * conf.getParameter<float>("timeResolutionFloor")},
      cone_{conf.getParameter<float>("maxBackwardDistance"),
            conf.getParameter<float>("maxLongitudinalDistance"),
            conf.getParameter<float>("transverseRadius0"),
            conf.getParameter<float>("transverseSlope")},
      emCone_{cone_.maxBackward,
              cone_.maxLongitudinal,
              conf.getParameter<float>("emTransverseRadius0"),
              conf.getParameter<float>("emTransverseSlope")},
      emCategories_(conf.getParameter<std::vector<int>>("emCategories")),
      emission_{conf.getParameter<float>("minEmittedEnergy"),
                conf.getParameter<float>("minEmittedPt"),
                conf.getParameter<float>("minEmittedEnergyEM"),
                conf.getParameter<float>("minEmittedPtEM"),
                conf.getParameter<float>("minEmittedEnergyAnyPt"),
                cuts_.typeVetoProbability},
      mergeFragments_(conf.getParameter<bool>("mergeFragments")),
      mergeDropped_(conf.getParameter<bool>("mergeDropped")),
      mergeHadronicOnly_(conf.getParameter<bool>("mergeHadronicOnly")),
      mergeMinParentEnergy_(conf.getParameter<float>("mergeMinParentEnergy")),
      mergeCone_{conf.getParameter<float>("mergeMaxBackwardDistance"),
                 conf.getParameter<float>("mergeMaxLongitudinalDistance"),
                 conf.getParameter<float>("mergeTransverseRadius0"),
                 conf.getParameter<float>("mergeTransverseSlope")},
      mergeEMCone_{mergeCone_.maxBackward,
                   mergeCone_.maxLongitudinal,
                   conf.getParameter<float>("mergeEMTransverseRadius0"),
                   conf.getParameter<float>("mergeEMTransverseSlope")} {
  const float etaWindow = cuts_.etaWindow;
  const float veto = cuts_.typeVetoProbability;
  const float timeFloor = conf.getParameter<float>("timeResolutionFloor");
  // Keeps the two ends of the phi search box of the tile in different bins.
  constexpr float kMaxEtaWindow = std::numbers::pi_v<float> * (1.f - 2.f / TileConstants::nPhiBins);
  const bool vetoValid = veto == 0.f || (veto >= 0.5f && veto <= 1.f);
  auto sortedCategories = emCategories_;
  std::sort(sortedCategories.begin(), sortedCategories.end());
  const bool categoriesValid =
      !sortedCategories.empty() && sortedCategories.front() >= 0 &&
      sortedCategories.back() < cornetto::kNumParticleTypes &&
      std::adjacent_find(sortedCategories.begin(), sortedCategories.end()) == sortedCategories.end();
  const auto coneValid = [](const cornetto::Cone& c, const cornetto::Cone& em) {
    return c.maxLongitudinal > 0.f && c.maxBackward >= 0.f && c.radius0 >= 0.f && c.slope >= 0.f && em.radius0 >= 0.f &&
           em.radius0 <= c.radius0 && em.slope >= 0.f && em.slope <= c.slope;
  };
  if (!(etaWindow > 0.f && etaWindow < kMaxEtaWindow) || !coneValid(cone_, emCone_) ||
      !coneValid(mergeCone_, mergeEMCone_) || !(timeFloor >= 0.f) || !vetoValid || !categoriesValid) {
    cms::Exception e("Configuration");
    e << "TracksterLinkingbyCornetto: invalid parameters, see the parameter descriptions. etaWindow " << etaWindow
      << ", maxLongitudinalDistance " << cone_.maxLongitudinal << ", maxBackwardDistance " << cone_.maxBackward
      << ", transverseRadius0 " << cone_.radius0 << ", transverseSlope " << cone_.slope << ", emTransverseRadius0 "
      << emCone_.radius0 << ", emTransverseSlope " << emCone_.slope << ", timeResolutionFloor " << timeFloor
      << ", typeVetoProbability " << veto << ", merge cone " << mergeCone_.maxBackward << " "
      << mergeCone_.maxLongitudinal << " " << mergeCone_.radius0 << " " << mergeCone_.slope << ", merge EM cone "
      << mergeEMCone_.radius0 << " " << mergeEMCone_.slope << ", emCategories";
    for (int c : emCategories_)
      e << " " << c;
    throw e;
  }
}

TracksterLinkingbyCornetto::Features TracksterLinkingbyCornetto::computeFeatures(const Inputs& input) const {
  const auto& tracksters = input.tracksters;
  const std::size_t n = tracksters.size();
  Features f;
  f.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    auto const& ts = tracksters[i];
    const auto& b = ts.barycenter();
    f.bary[i] = b;
    f.energy[i] = ts.raw_energy();
    f.valid[i] = std::isfinite(b.x()) && std::isfinite(b.y()) && std::isfinite(b.z()) && std::isfinite(f.energy[i]);
    f.eta[i] = b.eta();
    f.phi[i] = b.phi();
    f.rad[i] = std::sqrt(b.Mag2());
    f.time[i] = ts.time();
    f.timeError[i] = ts.timeError();
    f.em[i] = cornetto::emProbability(ts, emCategories_);
  }
  return f;
}

bool TracksterLinkingbyCornetto::mergeCompatible(const Features& g, unsigned int a, unsigned int b) const {
  const float veto = cuts_.typeVetoProbability;
  if (mergeHadronicOnly_ && veto > 0.f && (g.em[a] > veto || g.em[b] > veto))
    return false;
  if (!(g.energy[a] >= mergeMinParentEnergy_))
    return false;
  return cornetto::compatible(cuts_, mergeCone_, mergeEMCone_, g, a, b);
}

std::vector<std::vector<unsigned int>> TracksterLinkingbyCornetto::mergeGroups(
    const Features& f, std::vector<std::vector<unsigned int>> groups) const {
  if (!mergeFragments_ && !mergeDropped_)
    return groups;
  const Features g = cornetto::groupFeatures(f, groups);
  const auto n = static_cast<unsigned int>(groups.size());
  std::vector<bool> emitted(n);
  for (unsigned int k = 0; k < n; ++k)
    emitted[k] = emits(g, k);
  const auto tiles = std::make_unique<cornetto::EndcapTiles>();
  cornetto::fillTiles(g, *tiles);
  const auto allowed = [&](unsigned int a, unsigned int b) {
    return emitted[a] && (emitted[b] ? mergeFragments_ : mergeDropped_);
  };
  const auto test = [&](unsigned int a, unsigned int b) { return mergeCompatible(g, a, b); };
  const auto merged =
      cornetto::groups(cornetto::roots(cornetto::findParents(g, *tiles, mergeCone_, cuts_.etaWindow, allowed, test)),
                       [&](unsigned int root, unsigned int k) { return test(root, k); });
  std::vector<std::vector<unsigned int>> out;
  out.reserve(merged.size());
  for (auto const& m : merged) {
    std::vector<unsigned int> members;
    for (unsigned int k : m)
      members.insert(members.end(), groups[k].begin(), groups[k].end());
    std::sort(members.begin(), members.end());
    out.push_back(std::move(members));
  }
  std::sort(out.begin(), out.end(), [](auto const& x, auto const& y) { return x.front() < y.front(); });
  return out;
}

void TracksterLinkingbyCornetto::linkTracksters(
    const Inputs& input,
    std::vector<Trackster>& resultTracksters,
    std::vector<std::vector<unsigned int>>& linkedResultTracksters,
    std::vector<std::vector<unsigned int>>& linkedTracksterIdToInputTracksterId) {
  const Features f = computeFeatures(input);

  // About 200 kB, so on the heap rather than on the stack.
  const auto tiles = std::make_unique<cornetto::EndcapTiles>();
  cornetto::fillTiles(f, *tiles);

  // A trackster that is not compatible with its root leaves it.
  const auto test = [&](unsigned int a, unsigned int b) { return compatible(f, a, b); };
  const auto parents =
      cornetto::findParents(f, *tiles, cone_, cuts_.etaWindow, [](unsigned int, unsigned int) { return true; }, test);
  auto groups = mergeGroups(f, cornetto::groups(cornetto::roots(parents), test));

  // The emission rule reads the group energy, the pt along the energy-weighted member barycenter and the group EM
  // probability.
  const Features g = cornetto::groupFeatures(f, groups);
  resultTracksters.reserve(resultTracksters.size() + groups.size());
  linkedResultTracksters.reserve(linkedResultTracksters.size() + groups.size());
  linkedTracksterIdToInputTracksterId.reserve(linkedTracksterIdToInputTracksterId.size() + groups.size());
  for (unsigned int k = 0; k < groups.size(); ++k) {
    if (!emits(g, k))
      continue;
    Trackster merged;
    merged.mergeTracksters(input.tracksters, groups[k]);
    linkedResultTracksters.push_back({static_cast<unsigned int>(resultTracksters.size())});
    resultTracksters.push_back(std::move(merged));
    linkedTracksterIdToInputTracksterId.push_back(std::move(groups[k]));
  }
}
