#include <algorithm>
#include "FWCore/Utilities/interface/Exception.h"

#include "CondFormats/HcalObjects/interface/HcalCubicInterpolator.h"

HcalCubicInterpolator::HcalCubicInterpolator() {}

HcalCubicInterpolator::HcalCubicInterpolator(const std::vector<Triple>& points) {
  const std::size_t sz = points.size();
  if (sz) {
    std::vector<Triple> tmp(points);
    std::sort(tmp.begin(), tmp.end());
    abscissae_.reserve(sz);
    values_.reserve(sz);
    derivatives_.reserve(sz);
    for (std::size_t i = 0; i < sz; ++i) {
      const Triple& t(tmp[i]);
      abscissae_.push_back(std::get<0>(t));
      values_.push_back(std::get<1>(t));
      derivatives_.push_back(std::get<2>(t));
    }
    const std::size_t szm1 = sz - 1;
    for (std::size_t i = 0; i < szm1; ++i)
      if (abscissae_[i] == abscissae_[i + 1])
        throw cms::Exception(
            "In HcalCubicInterpolator constructor:"
            " abscissae must not coincide");
  }
}

double HcalCubicInterpolator::operator()(const double x) const {
  double result = 0.0;
  const std::size_t sz = abscissae_.size();
  if (sz) {
    if (sz > 1) {
      const std::size_t szm1 = sz - 1;
      if (x >= abscissae_[szm1])
        result = values_[szm1] + derivatives_[szm1] * (x - abscissae_[szm1]);
      else if (x <= abscissae_[0])
        result = values_[0] + derivatives_[0] * (x - abscissae_[0]);
      else {
        const std::size_t cell = std::upper_bound(abscissae_.begin(), abscissae_.end(), x) - abscissae_.begin() - 1;
        const std::size_t cellp1 = cell + 1;
        const double dx = abscissae_[cellp1] - abscissae_[cell];
        const double t = (x - abscissae_[cell]) / dx;
        const double onemt = 1.0 - t;
        const double h00 = onemt * onemt * (1.0 + 2.0 * t);
        const double h10 = onemt * onemt * t;
        const double h01 = t * t * (3.0 - 2.0 * t);
        const double h11 = t * t * onemt;
        result = h00 * values_[cell] + h10 * dx * derivatives_[cell] + h01 * values_[cellp1] -
                 h11 * dx * derivatives_[cellp1];
      }
    } else
      result = values_[0] + derivatives_[0] * (x - abscissae_[0]);
  }
  return result;
}

double HcalCubicInterpolator::xmin() const {
  double result = 0.0;
  if (!abscissae_.empty())
    result = abscissae_[0];
  return result;
}

double HcalCubicInterpolator::xmax() const {
  double result = 0.0;
  if (!abscissae_.empty())
    result = abscissae_.back();
  return result;
}

HcalCubicInterpolator HcalCubicInterpolator::approximateInverse() const {
  const bool monotonous =
      isStrictlyIncreasing(values_.begin(), values_.end()) || isStrictlyDecreasing(values_.begin(), values_.end());
  if (!monotonous)
    throw cms::Exception(
        "In HcalCubicInterpolator::inverse:"
        " can't invert non-monotonous functor");
  const std::size_t sz = abscissae_.size();
  std::vector<Triple> points;
  points.reserve(sz);
  for (std::size_t i = 0; i < sz; ++i) {
    const double dydx = derivatives_[i];
    if (dydx == 0.0)
      throw cms::Exception(
          "In HcalCubicInterpolator::inverse:"
          " can't invert functor with derivatives of 0");
    points.push_back(Triple(values_[i], abscissae_[i], 1.0 / dydx));
  }
  return HcalCubicInterpolator(points);
}

BOOST_CLASS_EXPORT_IMPLEMENT(HcalCubicInterpolator)
