#include <algorithm>
#include "FWCore/Utilities/interface/Exception.h"

#include "CondFormats/HcalObjects/interface/HcalInterpolatedTableFunctor.h"

inline static double interpolateStep(
    const double x0, const double step, const double y0, const double y1, const double x) {
  return y0 + (y1 - y0) * ((x - x0) / step);
}

HcalInterpolatedTableFunctor::HcalInterpolatedTableFunctor()
    : xmin_(0.), xmax_(0.), leftExtrapolationLinear_(false), rightExtrapolationLinear_(false) {}

HcalInterpolatedTableFunctor::HcalInterpolatedTableFunctor(const std::vector<double>& values,
                                                           const double ixmin,
                                                           const double ixmax,
                                                           const bool leftExtrapolationLinear,
                                                           const bool rightExtrapolationLinear)
    : values_(values),
      xmin_(ixmin),
      xmax_(ixmax),
      leftExtrapolationLinear_(leftExtrapolationLinear),
      rightExtrapolationLinear_(rightExtrapolationLinear) {
  if (values_.size() < 2)
    throw cms::Exception(
        "In HcalInterpolatedTableFunctor constructor:"
        " insufficient number of points");
  if (xmin_ >= xmax_)
    throw cms::Exception(
        "In HcalInterpolatedTableFunctor constructor:"
        " invalid min and/or max coordinates");
}

double HcalInterpolatedTableFunctor::operator()(const double x) const {
  double result = 0.0;
  const std::size_t sz = values_.size();
  const std::size_t szm1 = sz - 1;
  const double step = (xmax_ - xmin_) / szm1;

  if (x >= xmax_) {
    if (rightExtrapolationLinear_)
      result = interpolateStep(xmax_ - step, step, values_[sz - 2], values_[szm1], x);
    else
      result = values_[szm1];
  } else if (x <= xmin_) {
    if (leftExtrapolationLinear_)
      result = interpolateStep(xmin_, step, values_[0], values_[1], x);
    else
      result = values_[0];
  } else {
    const std::size_t ux = static_cast<std::size_t>((x - xmin_) / step);
    if (ux >= szm1)
      return values_[szm1];
    result = interpolateStep(ux * step + xmin_, step, values_[ux], values_[ux + 1], x);
  }

  return result;
}

bool HcalInterpolatedTableFunctor::isStrictlyMonotonous() const {
  return isStrictlyIncreasing(values_.begin(), values_.end()) || isStrictlyDecreasing(values_.begin(), values_.end());
}

HcalPiecewiseLinearFunctor HcalInterpolatedTableFunctor::inverse() const {
  if (!isStrictlyMonotonous())
    throw cms::Exception(
        "In HcalInterpolatedTableFunctor::inverse:"
        " can't invert non-monotonous functor");
  const std::size_t sz = values_.size();
  const std::size_t szm1 = sz - 1;
  const double step = (xmax_ - xmin_) / szm1;
  std::vector<std::pair<double, double> > points;
  points.reserve(sz);
  for (std::size_t i = 0; i < sz; ++i) {
    const double x = (i == szm1 ? xmax_ : xmin_ + step * i);
    points.push_back(std::make_pair(values_[i], x));
  }
  bool l = leftExtrapolationLinear_;
  bool r = rightExtrapolationLinear_;
  if (values_[0] > values_[sz - 1])
    std::swap(l, r);
  return HcalPiecewiseLinearFunctor(points, l, r);
}

BOOST_CLASS_EXPORT_IMPLEMENT(HcalInterpolatedTableFunctor)
