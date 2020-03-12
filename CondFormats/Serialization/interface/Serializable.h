#pragma once

#if defined(__GCCXML__)

#define COND_SERIALIZABLE
#define COND_TRANSIENT

#else

// The archives must be listed before any boost/serialization header.
// Otherwise, in some cases the export macros trigger compilation errors.
// #include "CondFormats/Serialization/interface/Archive.h"

#include <boost/serialization/access.hpp>

#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/bitset.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/utility.hpp>

// We cannot include Equal.h here since it is C++11
namespace cond {
  namespace serialization {
    template <typename CondSerializationT, typename Enabled = void>
    struct access;
  }
}  // namespace cond

// Marks a class/struct as serializable Conditions.
// It must be used in the end of the class/struct, to avoid
// changing the default access specifier.
// Note: the serialization code generator script depends on
//       the implementation of the macro.
#define COND_SERIALIZABLE                                  \
private:                                                   \
  template <class Archive>                                 \
  void serialize(Archive& ar, const unsigned int version); \
  template <typename CondSerializationT, typename Enabled> \
  friend struct cond::serialization::access;               \
  friend class boost::serialization::access;

// Same, but does *not* automatically generate the serialization code.
// This is useful when special features are required, e.g. versioning
// or using non-deducible contexts.
#define COND_SERIALIZABLE_MANUAL \
  COND_SERIALIZABLE;             \
  void cond_serialization_manual();

// Polymorphic classes must be tagged as such
#define COND_SERIALIZABLE_POLYMORPHIC(T) BOOST_CLASS_EXPORT(T);

// Marks a member as transient, i.e. not included in the automatically
// generated serialization code. All variables in the same 'statement'
// (up to the ';') will be marked as transient, so please avoid declaring
// more than one transient member per 'statement'/line. In order to
// avoid that, in the future we may be able to use custom C++11 attributes
// like [[cond::serialization::transient]]
#define COND_TRANSIENT

#endif /* !defined(__GCCXML__) */
