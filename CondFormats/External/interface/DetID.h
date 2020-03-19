#ifndef CondFormats_External_DETID_H
#define CondFormats_External_DETID_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_free.hpp>

// std::vector used in DataFormats/EcalDetId/interface/EcalContainer.h
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/map.hpp>

#include "DataFormats/DetId/interface/DetId.h"

namespace boost {
  namespace serialization {

    /*
 * Note regarding object tracking: all autos used here
 * must resolve to untracked types, since we use local
 * variables in the stack which could end up with the same
 * address. For the moment, all types resolved by auto here
 * are primitive types, which are untracked by default
 * by Boost Serialization.
 */

    // DataFormats/DetId/interface/DetId.h
    template <class Archive>
    void save(Archive& ar, const DetId& obj, const unsigned int) {
      auto id_ = obj.rawId();
      ar& BOOST_SERIALIZATION_NVP(id_);
    }

    template <class Archive>
    void load(Archive& ar, DetId& obj, const unsigned int) {
      decltype(obj.rawId()) id_;
      ar& BOOST_SERIALIZATION_NVP(id_);
      obj = DetId(id_);
    }

    template <class Archive>
    void serialize(Archive& ar, DetId& obj, const unsigned int v) {
      split_free(ar, obj, v);
    }

  }  // namespace serialization
}  // namespace boost

#endif
