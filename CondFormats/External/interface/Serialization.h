#ifndef CondFormats_External_Serialization_H
#define CondFormats_External_Serialization_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_free.hpp>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EcalContainer.h"
#include "DataFormats/HLTReco/interface/HLTPrescaleTable.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

#include "CLHEP/Vector/EulerAngles.h"
#include "CLHEP/Vector/ThreeVector.h"

#include <Math/SMatrix.h>

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
template<class Archive>
void serialize(Archive & ar, DetId & obj, const unsigned int)
{
    auto id_ = obj.rawId();
    ar & BOOST_SERIALIZATION_NVP(id_);
    obj = DetId(id_);
}

// DataFormats/EcalDetId/interface/EBDetId.h
template<class Archive>
void serialize(Archive & ar, EBDetId & obj, const unsigned int)
{
    // TODO
}

// DataFormats/EcalDetId/interface/EcalContainer.h
template<class Archive, typename DetIdT, typename T>
void serialize(Archive & ar, EcalContainer<DetIdT, T> & obj, const unsigned int)
{
    // TODO
}

// DataFormats/HLTReco/interface/HLTPrescaleTable.h
template<class Archive>
void serialize(Archive & ar, trigger::HLTPrescaleTable & obj, const unsigned int)
{
    // TODO
}

// DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h
template<class Archive>
void serialize(Archive & ar, L1GtLogicParser::TokenRPN & obj, const unsigned int)
{
    // TODO
}

// DataFormats/Provenance/interface/Timestamp.h
template<class Archive>
void serialize(Archive & ar, edm::Timestamp & obj, const unsigned int)
{
    // TODO
}

// CLHEP/Vector/ThreeVector.h
template<class Archive>
void serialize(Archive & ar, CLHEP::Hep3Vector & obj, const unsigned int)
{
    auto dx = obj.x();
    auto dy = obj.y();
    auto dz = obj.z();
    ar & BOOST_SERIALIZATION_NVP(dx);
    ar & BOOST_SERIALIZATION_NVP(dy);
    ar & BOOST_SERIALIZATION_NVP(dz);
    obj.setX(dx);
    obj.setY(dy);
    obj.setZ(dz);
}

// CLHEP/Vector/EulerAngles.h
template<class Archive>
void serialize(Archive & ar, CLHEP::HepEulerAngles & obj, const unsigned int)
{
    auto phi_ = obj.phi();
    auto theta_ = obj.theta();
    auto psi_ = obj.psi();
    ar & BOOST_SERIALIZATION_NVP(phi_);
    ar & BOOST_SERIALIZATION_NVP(theta_);
    ar & BOOST_SERIALIZATION_NVP(psi_);
    obj.setPhi(phi_);
    obj.setTheta(theta_);
    obj.setPsi(psi_);
}

// Math/SMatrix.h
template<class Archive, typename T, unsigned int D1, unsigned int D2, class R>
void serialize(Archive & ar, ROOT::Math::SMatrix<T, D1, D2, R> & obj, const unsigned int)
{
    unsigned int i = 0;
    for (auto & value : obj) {
        ar & boost::serialization::make_nvp(std::to_string(i).c_str(), value);
        ++i;
    }
}

} // namespace serialization
} // namespace boost


namespace cond {
namespace serialization {

// DataFormats/DetId/interface/DetId.h
template <>
struct access<DetId>
{
    static bool equal_(const DetId & first, const DetId & second)
    {
        return true
            and (equal(first.rawId(), second.rawId()))
        ;
    }
};

// DataFormats/EcalDetId/interface/EBDetId.h
template <>
struct access<EBDetId>
{
    static bool equal_(const EBDetId & first, const EBDetId & second)
    {
        return true
            // TODO
        ;
    }
};

// DataFormats/EcalDetId/interface/EcalContainer.h
template <typename DetIdT, typename T>
struct access<EcalContainer<DetIdT, T>>
{
    static bool equal_(const EcalContainer<DetIdT, T> & first, const EcalContainer<DetIdT, T> & second)
    {
        return true
            // TODO
        ;
    }
};


// DataFormats/HLTReco/interface/HLTPrescaleTable.h
template <>
struct access<trigger::HLTPrescaleTable>
{
    static bool equal_(const trigger::HLTPrescaleTable & first, const trigger::HLTPrescaleTable & second)
    {
        return true
            // TODO
        ;
    }
};

// DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h
template <>
struct access<L1GtLogicParser::TokenRPN>
{
    static bool equal_(const L1GtLogicParser::TokenRPN & first, const L1GtLogicParser::TokenRPN & second)
    {
        return true
            // TODO
        ;
    }
};

// DataFormats/Provenance/interface/Timestamp.h
template <>
struct access<edm::Timestamp>
{
    static bool equal_(const edm::Timestamp & first, const edm::Timestamp & second)
    {
        return true
            // TODO
        ;
    }
};

// CLHEP/Vector/ThreeVector.h
template <>
struct access<CLHEP::Hep3Vector>
{
    static bool equal_(const CLHEP::Hep3Vector & first, const CLHEP::Hep3Vector & second)
    {
        return true
            and (equal(first.x(), second.x()))
            and (equal(first.y(), second.y()))
            and (equal(first.z(), second.z()))
        ;
    }
};

// CLHEP/Vector/EulerAngles.h
template <>
struct access<CLHEP::HepEulerAngles>
{
    static bool equal_(const CLHEP::HepEulerAngles & first, const CLHEP::HepEulerAngles & second)
    {
        return true
            and (equal(first.phi(), second.phi()))
            and (equal(first.theta(), second.theta()))
            and (equal(first.psi(), second.psi()))
        ;
    }
};

// Math/SMatrix.h
template <typename T, unsigned int D1, unsigned int D2, class R>
struct access<ROOT::Math::SMatrix<T, D1, D2, R>>
{
    static bool equal_(const ROOT::Math::SMatrix<T, D1, D2, R> & first, const ROOT::Math::SMatrix<T, D1, D2, R> & second)
    {
        return true
            // TODO
        ;
    }
};

} // namespace serialization
} // namespace cond

#endif
