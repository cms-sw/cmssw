#ifndef RecoTracker_DeDx_LikelihoodFitDeDxEstimator_h
#define RecoTracker_DeDx_LikelihoodFitDeDxEstimator_h

#include <cmath>
#include "RecoTracker/DeDx/interface/BaseDeDxEstimator.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"

class LikelihoodFitDeDxEstimator : public BaseDeDxEstimator {
public:
  LikelihoodFitDeDxEstimator(const edm::ParameterSet& iConfig) {}

  std::pair<float, float> dedx(const reco::DeDxHitCollection& Hits) override {
    if (Hits.empty())
      return {0., 0.};
    // compute original
    std::array<double, 2> value;
    const auto& chi2 = estimate(Hits, value);
    // try to remove lowest dE/dx measurement
    const auto& n = Hits.size();
    if (n >= 3 && (chi2 > 1.3 * n + 4 * std::sqrt(1.3 * n))) {
      auto hs = Hits;
      hs.erase(std::min_element(hs.begin(), hs.end()));
      // if got better, accept
      std::array<double, 2> v;
      if (estimate(hs, v) < chi2 - 12)
        value = v;
    }
    return {value[0], std::sqrt(value[1])};
  }

private:
  void calculate_wrt_epsilon(const reco::DeDxHit&, const double&, std::array<double, 3>&);
  void functionEpsilon(const reco::DeDxHitCollection&, const double&, std::array<double, 3>&);
  double minimizeAllSaturated(const reco::DeDxHitCollection&, std::array<double, 2>&);
  double newtonMethodEpsilon(const reco::DeDxHitCollection&, std::array<double, 2>&);
  double estimate(const reco::DeDxHitCollection&, std::array<double, 2>&);
};

/*****************************************************************************/
inline void LikelihoodFitDeDxEstimator::calculate_wrt_epsilon(const reco::DeDxHit& h,
                                                              const double& epsilon,
                                                              std::array<double, 3>& value) {
  const auto& ls = h.pathLength();
  const auto& sn = h.error();      // energy sigma
  const auto y = h.charge() * ls;  // = g * y
  const auto sD = 2.E-3 + 0.095 * y;
  const auto ss = sD * sD + sn * sn;
  const auto s = std::sqrt(ss);
  const auto delta = epsilon * ls;
  const auto dy = delta - y;
  constexpr double nu(0.65);

  // calculate derivatives with respect to delta
  std::array<double, 3> val{{0.}};
  if (h.rawDetId() == 0) {  // normal
    if (dy < -nu * s) {
      val[0] = -2. * nu * dy / s - nu * nu;
      val[1] = -2. * nu / s;
      val[2] = 0.;
    } else {
      val[0] = dy * dy / ss;
      val[1] = 2. * dy / ss;
      val[2] = 2. / ss;
    }
  } else {  // saturated
    if (dy < s) {
      val[0] = -dy / s + 1.;
      val[1] = -1. / s;
      val[2] = 0.;
    } else {
      val[0] = 0.;
      val[1] = 0.;
      val[2] = 0.;
    }
  }

  // d/d delta -> d/d epsilon
  val[1] *= ls;
  val[2] *= ls * ls;

  // sum
  for (size_t k = 0; k < value.size(); k++)
    value[k] += val[k];
}

/*****************************************************************************/
inline void LikelihoodFitDeDxEstimator::functionEpsilon(const reco::DeDxHitCollection& Hits,
                                                        const double& epsilon,
                                                        std::array<double, 3>& val) {
  val = {{0, 0, 0}};
  for (const auto& h : Hits)
    calculate_wrt_epsilon(h, epsilon, val);
}

/*****************************************************************************/
inline double LikelihoodFitDeDxEstimator::minimizeAllSaturated(const reco::DeDxHitCollection& Hits,
                                                               std::array<double, 2>& value) {
  int nStep(0);
  double par(3.0);  // input MeV/cm

  std::array<double, 3> val{{0}};
  do {
    functionEpsilon(Hits, par, val);
    if (val[1] != 0)
      par += -val[0] / val[1];
    nStep++;
  } while (val[0] > 1e-3 && val[1] != 0 && nStep < 10);

  value[0] = par * 1.1;
  value[1] = par * par * 0.01;

  return val[0];
}

/*****************************************************************************/
inline double LikelihoodFitDeDxEstimator::newtonMethodEpsilon(const reco::DeDxHitCollection& Hits,
                                                              std::array<double, 2>& value) {
  int nStep(0);
  double par(3.0);  // input MeV/cm
  double dpar(0);

  std::array<double, 3> val{{0}};
  do {
    functionEpsilon(Hits, par, val);
    if (val[2] != 0.)
      dpar = -val[1] / std::abs(val[2]);
    else
      dpar = 1.;  // step up, for epsilon
    if (par + dpar > 0)
      par += dpar;  // ok
    else
      par /= 2.;  // half
    nStep++;
  } while (std::abs(dpar) > 1e-3 && nStep < 50);

  value[0] = par;
  value[1] = 2. / val[2];

  return val[0];
}

/*****************************************************************************/
inline double LikelihoodFitDeDxEstimator::estimate(const reco::DeDxHitCollection& Hits, std::array<double, 2>& value) {
  // use newton method if at least one hit is not saturated
  for (const auto& h : Hits)
    if (h.rawDetId() == 0)
      return newtonMethodEpsilon(Hits, value);
  // else use minimize all saturated
  return minimizeAllSaturated(Hits, value);
}

#endif
