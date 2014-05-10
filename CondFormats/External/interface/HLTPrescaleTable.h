#ifndef CondFormats_External_HLTPRESCALETABLE_H
#define CondFormats_External_HLTPRESCALETABLE_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_free.hpp>

// std::vector used in DataFormats/EcalDetId/interface/EcalContainer.h
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/map.hpp>

#include "DataFormats/HLTReco/interface/HLTPrescaleTable.h"

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

// DataFormats/HLTReco/interface/HLTPrescaleTable.h
template<class Archive>
void save(Archive & ar, const trigger::HLTPrescaleTable & obj, const unsigned int)
{
    auto set = obj.set();
    auto lab = obj.labels();
    auto tab = obj.table();
    ar & boost::serialization::make_nvp("set_"   , set );
    ar & boost::serialization::make_nvp("labels_", lab );
    ar & boost::serialization::make_nvp("table_" , tab );  
}

template<class Archive>
void load(Archive & ar, trigger::HLTPrescaleTable & obj, const unsigned int)
{
    // FIXME: avoid copying if we are OK getting a non-const reference
    unsigned int set_;
    std::vector<std::string> labels_;
    std::map<std::string,std::vector<unsigned int> > table_;

    ar & boost::serialization::make_nvp("set_"   , set_ );
    ar & boost::serialization::make_nvp("labels_", labels_ );
    ar & boost::serialization::make_nvp("table_" , table_ );  
    trigger::HLTPrescaleTable tmp(set_, labels_, table_);
    obj = tmp;

}

template<class Archive>
void serialize(Archive & ar, trigger::HLTPrescaleTable & obj, const unsigned int v)
{
    split_free(ar, obj, v);
}

} // namespace serialization
} // namespace boost

#endif
