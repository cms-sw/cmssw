#ifndef CondFormats_HcalObjects_HFPhase1PMTData_h
#define CondFormats_HcalObjects_HFPhase1PMTData_h

#include "boost/serialization/access.hpp"
#include "boost/serialization/version.hpp"
#include "boost/serialization/shared_ptr.hpp"
#include "boost/version.hpp"
#if BOOST_VERSION < 106400
#include "boost/serialization/array.hpp"
#else
#include "boost/serialization/boost_array.hpp"
#endif
#include "CondFormats/HcalObjects/interface/AbsHcalFunctor.h"

class HFPhase1PMTData {
public:
  // Functor enum for the cut shapes
  enum {
    T_0_MIN = 0,  // Min time measurement allowed for the first anode,
                  //    depending on charge or energy
    T_0_MAX,      // Max time measurement allowed for the first anode
    T_1_MIN,      // Min time measurement allowed for the second anode
    T_1_MAX,      // Max time measurement allowed for the second anode
    ASYMM_MIN,    // Minimum allowed charge (or energy) asymmetry,
                  //    depending on charge (or energy)
    ASYMM_MAX,    // Maximum allowed asymmetry
    N_PMT_CUTS
  };
  typedef boost::array<std::shared_ptr<AbsHcalFunctor>, N_PMT_CUTS> Cuts;

  // Dummy constructor, to be used for deserialization only
  inline HFPhase1PMTData() : minCharge0_(0.0), minCharge1_(0.0), minChargeAsymm_(0.0) {}

  // Normal constructor
  inline HFPhase1PMTData(const Cuts& cutShapes, const float charge0, const float charge1, const float minQAsymm)
      : cuts_(cutShapes), minCharge0_(charge0), minCharge1_(charge1), minChargeAsymm_(minQAsymm) {}

  // Get the cut shape
  inline const AbsHcalFunctor& cut(const unsigned which) const { return *cuts_.at(which); }

  // Minimum charge on the first/second anode needed for
  // a reliable timing measurement. Setting this charge
  // to a very high value will disable timing measurements
  // on that anode.
  inline float minCharge0() const { return minCharge0_; }
  inline float minCharge1() const { return minCharge1_; }

  // Minimum total charge for applying the charge asymmetry cut
  inline float minChargeAsymm() const { return minChargeAsymm_; }

  // Deep comparison operators (useful for serialization tests)
  inline bool operator==(const HFPhase1PMTData& r) const {
    if (minCharge0_ != r.minCharge0_)
      return false;
    if (minCharge1_ != r.minCharge1_)
      return false;
    if (minChargeAsymm_ != r.minChargeAsymm_)
      return false;
    for (unsigned i = 0; i < N_PMT_CUTS; ++i)
      if (!(*cuts_[i] == *r.cuts_[i]))
        return false;
    return true;
  }

  inline bool operator!=(const HFPhase1PMTData& r) const { return !(*this == r); }

private:
  Cuts cuts_;
  float minCharge0_;
  float minCharge1_;
  float minChargeAsymm_;

  friend class boost::serialization::access;

  template <class Archive>
  inline void serialize(Archive& ar, unsigned /* version */) {
    ar& cuts_& minCharge0_& minCharge1_& minChargeAsymm_;
  }
};

BOOST_CLASS_VERSION(HFPhase1PMTData, 1)

#endif  // CondFormats_HcalObjects_HFPhase1PMTData_h
