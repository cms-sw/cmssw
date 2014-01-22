#ifndef CondFormats_External_ECALDETID_H
#define CondFormats_External_ECALDETID_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_free.hpp>

// std::vector used in DataFormats/EcalDetId/interface/EcalContainer.h
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/map.hpp>

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalContainer.h"

// for base class
#include "CondFormats/External/interface/DetID.h"

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

// DataFormats/EcalDetId/interface/EBDetId.h
template<class Archive>
void serialize(Archive & ar, EBDetId & obj, const unsigned int)
{
    ar & boost::serialization::make_nvp("DetId", boost::serialization::base_object<DetId>(obj));;
}

// DataFormats/EcalDetId/interface/EEDetId.h
template<class Archive>
void serialize(Archive & ar, EEDetId & obj, const unsigned int)
{
    ar & boost::serialization::make_nvp("DetId", boost::serialization::base_object<DetId>(obj));;
}


// DataFormats/EcalDetId/interface/EcalContainer.h
template<class Archive, typename DetIdT, typename T>
void save(Archive & ar, const EcalContainer<DetIdT, T> & obj, const unsigned int)
{
    ar & boost::serialization::make_nvp("m_items", obj.items());
}

template<class Archive, typename DetIdT, typename T>
void load(Archive & ar, EcalContainer<DetIdT, T> & obj, const unsigned int)
{
    // FIXME: avoid copying if we are OK getting a non-const reference
    typename EcalContainer<DetIdT, T>::Items m_items;
    ar & boost::serialization::make_nvp("m_items", m_items);
    obj.setItems(m_items);
}

template<class Archive, typename DetIdT, typename T>
void serialize(Archive & ar, EcalContainer<DetIdT, T> & obj, const unsigned int v)
{
    split_free(ar, obj, v);
}

} // namespace serialization
} // namespace boost

namespace cond {
namespace serialization {

// DataFormats/EcalDetId/interface/EBDetId.h
template <>
struct access<EBDetId>
{
    static bool equal_(const EBDetId & first, const EBDetId & second)
    {
        return true
            and (equal(static_cast<const DetId &>(first), static_cast<const DetId &>(second)))
        ;
    }
};
  
// DataFormats/EcalDetId/interface/EEDetId.h
template <>
struct access<EEDetId>
{
    static bool equal_(const EEDetId & first, const EEDetId & second)
    {
        return true
            and (equal(static_cast<const DetId &>(first), static_cast<const DetId &>(second)))
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
	  and (equal(first.items(), second.items()))
        ;
    }
};



} // namespace serialization
} // namespace cond

#endif
