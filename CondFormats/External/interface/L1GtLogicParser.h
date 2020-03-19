#ifndef CondFormats_External_L1GTLOGICPARSER_H
#define CondFormats_External_L1GTLOGICPARSER_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_free.hpp>

// std::vector used in DataFormats/EcalDetId/interface/EcalContainer.h
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/map.hpp>

#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"

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

    // DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h
    template <class Archive>
    void serialize(Archive& ar, L1GtLogicParser::TokenRPN& obj, const unsigned int) {
      ar& boost::serialization::make_nvp("operation", obj.operation);
      ar& boost::serialization::make_nvp("operand", obj.operand);
    }

  }  // namespace serialization
}  // namespace boost

#endif
