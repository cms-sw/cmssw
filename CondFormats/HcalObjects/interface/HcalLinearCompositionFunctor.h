#ifndef CondFormats_HcalObjects_HcalLinearCompositionFunctor_h
#define CondFormats_HcalObjects_HcalLinearCompositionFunctor_h

#include "FWCore/Utilities/interface/Exception.h"

#include "CondFormats/HcalObjects/interface/AbsHcalFunctor.h"

#include "boost/serialization/access.hpp"
#include "boost/serialization/version.hpp"
#include "boost/serialization/shared_ptr.hpp"

//
// A functor returning a linearly transformed value
// of another functor: f(x) = a*p(x) + b. Useful for
// implementing cuts symmetric about 0, etc.
//
class HcalLinearCompositionFunctor : public AbsHcalFunctor {
public:
  // Dummy constructor, to be used for deserialization only
  inline HcalLinearCompositionFunctor() : a_(0.0), b_(0.0) {}

  // Normal constructor
  HcalLinearCompositionFunctor(std::shared_ptr<AbsHcalFunctor> p, double a, double b);

  inline ~HcalLinearCompositionFunctor() override {}

  double operator()(double x) const override;

  inline double xmin() const override { return other_->xmin(); }
  inline double xmax() const override { return other_->xmax(); }

  inline double a() const { return a_; }
  inline double b() const { return b_; }

protected:
  inline bool isEqual(const AbsHcalFunctor& other) const override {
    const HcalLinearCompositionFunctor& r = static_cast<const HcalLinearCompositionFunctor&>(other);
    return *other_ == *r.other_ && a_ == r.a_ && b_ == r.b_;
  }

private:
  std::shared_ptr<AbsHcalFunctor> other_;
  double a_;
  double b_;

  friend class boost::serialization::access;

  template <class Archive>
  inline void serialize(Archive& ar, unsigned /* version */) {
    boost::serialization::base_object<AbsHcalFunctor>(*this);
    // Direct polymorphic serialization of shared_ptr is broken
    // in boost for versions 1.56, 1.57, 1.58. For detail, see
    // https://svn.boost.org/trac/boost/ticket/10727
#if BOOST_VERSION < 105600 || BOOST_VERSION > 105800
    ar& other_& a_& b_;
#else
    throw cms::Exception(
        "HcalLinearCompositionFunctor can not be"
        " serialized with this version of boost");
#endif
  }
};

BOOST_CLASS_VERSION(HcalLinearCompositionFunctor, 1)
BOOST_CLASS_EXPORT_KEY(HcalLinearCompositionFunctor)

#endif  // CondFormats_HcalObjects_HcalLinearCompositionFunctor_h
