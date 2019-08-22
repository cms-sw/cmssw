#ifndef CondFormats_External_CLHEP_H
#define CondFormats_External_CLHEP_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_free.hpp>

// std::vector used in DataFormats/EcalDetId/interface/EcalContainer.h
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/map.hpp>

#include "CLHEP/Vector/EulerAngles.h"
#include "CLHEP/Vector/ThreeVector.h"

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

    // CLHEP/Vector/ThreeVector.h
    template <class Archive>
    void save(Archive& ar, const CLHEP::Hep3Vector& obj, const unsigned int) {
      auto dx = obj.x();
      auto dy = obj.y();
      auto dz = obj.z();
      ar& BOOST_SERIALIZATION_NVP(dx);
      ar& BOOST_SERIALIZATION_NVP(dy);
      ar& BOOST_SERIALIZATION_NVP(dz);
    }

    template <class Archive>
    void load(Archive& ar, CLHEP::Hep3Vector& obj, const unsigned int) {
      decltype(obj.x()) dx;
      decltype(obj.y()) dy;
      decltype(obj.z()) dz;
      ar& BOOST_SERIALIZATION_NVP(dx);
      ar& BOOST_SERIALIZATION_NVP(dy);
      ar& BOOST_SERIALIZATION_NVP(dz);
      obj.set(dx, dy, dz);
    }

    template <class Archive>
    void serialize(Archive& ar, CLHEP::Hep3Vector& obj, const unsigned int v) {
      split_free(ar, obj, v);
    }

    // CLHEP/Vector/EulerAngles.h
    template <class Archive>
    void save(Archive& ar, const CLHEP::HepEulerAngles& obj, const unsigned int) {
      auto phi_ = obj.phi();
      auto theta_ = obj.theta();
      auto psi_ = obj.psi();
      ar& BOOST_SERIALIZATION_NVP(phi_);
      ar& BOOST_SERIALIZATION_NVP(theta_);
      ar& BOOST_SERIALIZATION_NVP(psi_);
    }

    template <class Archive>
    void load(Archive& ar, CLHEP::HepEulerAngles& obj, const unsigned int) {
      decltype(obj.phi()) phi_;
      decltype(obj.theta()) theta_;
      decltype(obj.psi()) psi_;
      ar& BOOST_SERIALIZATION_NVP(phi_);
      ar& BOOST_SERIALIZATION_NVP(theta_);
      ar& BOOST_SERIALIZATION_NVP(psi_);
      obj.set(phi_, theta_, psi_);
    }

    template <class Archive>
    void serialize(Archive& ar, CLHEP::HepEulerAngles& obj, const unsigned int v) {
      split_free(ar, obj, v);
    }

  }  // namespace serialization
}  // namespace boost

#endif
