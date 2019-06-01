#ifndef CondFormats_HcalObjects_AbsHcalFunctor_h
#define CondFormats_HcalObjects_AbsHcalFunctor_h

#include <cfloat>
#include <typeinfo>

#include "boost/serialization/access.hpp"
#include "boost/serialization/base_object.hpp"
#include "boost/serialization/export.hpp"

// Archive headers are needed here for the serialization registration to work.
// <cassert> is needed for the archive headers to work.
#if !defined(__GCCXML__)
#include <cassert>
#include "CondFormats/Serialization/interface/eos/portable_iarchive.hpp"
#include "CondFormats/Serialization/interface/eos/portable_oarchive.hpp"
#endif /* #if !defined(__GCCXML__) */

class AbsHcalFunctor {
public:
  inline virtual ~AbsHcalFunctor() {}

  // Method to override by concrete storable functor classes
  virtual double operator()(double x) const = 0;

  // Functor domain. Should be overriden by derived classes if needed.
  inline virtual double xmin() const { return -DBL_MAX; }
  inline virtual double xmax() const { return DBL_MAX; }

  // Comparison operators. Note that they are not virtual and should
  // not be overriden by derived classes. These operators are very
  // useful for I/O testing.
  inline bool operator==(const AbsHcalFunctor& r) const { return (typeid(*this) == typeid(r)) && this->isEqual(r); }
  inline bool operator!=(const AbsHcalFunctor& r) const { return !(*this == r); }

protected:
  // Method needed to compare objects for equality.
  // Must be implemented by derived classes.
  virtual bool isEqual(const AbsHcalFunctor&) const = 0;

  // Check if the sequence of values is strictly increasing
  template <class Iter>
  static bool isStrictlyIncreasing(Iter begin, Iter const end) {
    if (begin == end)
      return false;
    Iter first(begin);
    bool status = ++begin != end;
    for (; begin != end && status; ++begin, ++first)
      if (!(*first < *begin))
        status = false;
    return status;
  }

  // Check if the sequence of values is strictly decreasing
  template <class Iter>
  static bool isStrictlyDecreasing(Iter begin, Iter const end) {
    if (begin == end)
      return false;
    Iter first(begin);
    bool status = ++begin != end;
    for (; begin != end && status; ++begin, ++first)
      if (!(*begin < *first))
        status = false;
    return status;
  }

private:
  friend class boost::serialization::access;
  template <typename Ar>
  inline void serialize(Ar& ar, unsigned /* version */) {}
};

BOOST_CLASS_EXPORT_KEY(AbsHcalFunctor)

#endif  // CondFormats_HcalObjects_AbsHcalFunctor_h
