#include <algorithm>
#include "FWCore/Utilities/interface/Exception.h"

#include "CondFormats/HcalObjects/interface/HcalPiecewiseLinearFunctor.h"

inline static double interpolateSimple(
    const double x0, const double x1, const double y0, const double y1, const double x) {
  return y0 + (y1 - y0) * ((x - x0) / (x1 - x0));
}

HcalPiecewiseLinearFunctor::HcalPiecewiseLinearFunctor()
    : leftExtrapolationLinear_(false), rightExtrapolationLinear_(false) {}

HcalPiecewiseLinearFunctor::HcalPiecewiseLinearFunctor(const std::vector<std::pair<double, double> >& points,
                                                       const bool leftExtrapolationLinear,
                                                       const bool rightExtrapolationLinear)
    : leftExtrapolationLinear_(leftExtrapolationLinear), rightExtrapolationLinear_(rightExtrapolationLinear) {
  const std::size_t sz = points.size();
  if (sz) {
    std::vector<std::pair<double, double> > tmp(points);
    std::sort(tmp.begin(), tmp.end());
    abscissae_.reserve(sz);
    values_.reserve(sz);
    for (std::size_t i = 0; i < sz; ++i) {
      abscissae_.push_back(tmp[i].first);
      values_.push_back(tmp[i].second);
    }
    if (!isStrictlyIncreasing(abscissae_.begin(), abscissae_.end()))
      throw cms::Exception(
          "In HcalPiecewiseLinearFunctor constructor:"
          " abscissae must not coincide");
  }
}

double HcalPiecewiseLinearFunctor::operator()(const double x) const {
  double result = 0.0;
  const std::size_t sz = abscissae_.size();
  if (sz) {
    if (sz > 1) {
      const std::size_t szm1 = sz - 1;
      if (x >= abscissae_[szm1]) {
        if (rightExtrapolationLinear_)
          result = interpolateSimple(abscissae_[sz - 2], abscissae_[szm1], values_[sz - 2], values_[szm1], x);
        else
          result = values_[szm1];
      } else if (x <= abscissae_[0]) {
        if (leftExtrapolationLinear_)
          result = interpolateSimple(abscissae_[0], abscissae_[1], values_[0], values_[1], x);
        else
          result = values_[0];
      } else {
        const std::size_t cell = std::upper_bound(abscissae_.begin(), abscissae_.end(), x) - abscissae_.begin() - 1;
        result = interpolateSimple(abscissae_[cell], abscissae_[cell + 1], values_[cell], values_[cell + 1], x);
      }
    } else
      result = values_[0];
  }
  return result;
}

bool HcalPiecewiseLinearFunctor::isStrictlyMonotonous() const {
  return isStrictlyIncreasing(values_.begin(), values_.end()) || isStrictlyDecreasing(values_.begin(), values_.end());
}

HcalPiecewiseLinearFunctor HcalPiecewiseLinearFunctor::inverse() const {
  if (!isStrictlyMonotonous())
    throw cms::Exception(
        "In HcalPiecewiseLinearFunctor::inverse:"
        " can't invert non-monotonous functor");
  const std::size_t sz = abscissae_.size();
  std::vector<std::pair<double, double> > points;
  points.reserve(sz);
  for (std::size_t i = 0; i < sz; ++i)
    points.push_back(std::make_pair(values_[i], abscissae_[i]));
  bool l = leftExtrapolationLinear_;
  bool r = rightExtrapolationLinear_;
  if (values_[0] > values_[sz - 1])
    std::swap(l, r);
  return HcalPiecewiseLinearFunctor(points, l, r);
}

double HcalPiecewiseLinearFunctor::xmin() const {
  double result = 0.0;
  if (!abscissae_.empty())
    result = abscissae_[0];
  return result;
}

double HcalPiecewiseLinearFunctor::xmax() const {
  double result = 0.0;
  if (!abscissae_.empty())
    result = abscissae_.back();
  return result;
}

BOOST_CLASS_EXPORT_IMPLEMENT(HcalPiecewiseLinearFunctor)
