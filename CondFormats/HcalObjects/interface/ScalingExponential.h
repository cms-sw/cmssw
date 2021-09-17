#ifndef CondFormats_HcalObjects_ScalingExponential_h_
#define CondFormats_HcalObjects_ScalingExponential_h_

#include <cmath>

#include "boost/serialization/access.hpp"
#include "boost/serialization/version.hpp"
#include <cstdint>

class ScalingExponential {
public:
  inline ScalingExponential() : p0_(0.0), p1_(0.0), linear_(0) {}

  inline ScalingExponential(const double p0, const double p1, const bool isLinear = false)
      : p0_(p0), p1_(p1), linear_(isLinear) {}

  inline double operator()(const double x) const {
    const double scale = linear_ ? p0_ * x + p1_ : p0_ * (1.0 - exp(-x / p1_));
    return scale * x;
  }

  inline bool operator==(const ScalingExponential& r) const {
    return p0_ == r.p0_ && p1_ == r.p1_ && linear_ == r.linear_;
  }

  inline bool operator!=(const ScalingExponential& r) const { return !(*this == r); }

private:
  double p0_;
  double p1_;
  uint8_t linear_;

  friend class boost::serialization::access;

  template <class Archive>
  inline void serialize(Archive& ar, unsigned /* version */) {
    ar& p0_& p1_& linear_;
  }
};

BOOST_CLASS_VERSION(ScalingExponential, 1)

#endif  // CondFormats_HcalObjects_ScalingExponential_h_
