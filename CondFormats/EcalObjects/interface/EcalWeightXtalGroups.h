#ifndef CondFormats_EcalObjects_EcalWeightXtalGroups_H
#define CondFormats_EcalObjects_EcalWeightXtalGroups_H
/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Created: 22 Feb 2006
 * $Id: $
 **/

#include <map>
#include <boost/cstdint.hpp>
#include "CondFormats/EcalObjects/interface/EcalXtalGroupId.h"

class EcalWeightXtalGroups {
  public:
    typedef std::map<uint32_t, EcalXtalGroupId> EcalXtalGroupsMap;

    EcalWeightXtalGroups();
    ~EcalWeightXtalGroups();
    void  setValue(const uint32_t& xtal, const EcalXtalGroupId& group);
    const EcalXtalGroupsMap& getMap() const { return map_; }

  private:
    EcalXtalGroupsMap map_;
};
#endif
