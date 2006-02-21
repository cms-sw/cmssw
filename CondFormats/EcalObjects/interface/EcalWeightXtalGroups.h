#ifndef CondFormats_EcalObjects_EcalWeightXtalGroups_H
#define CondFormats_EcalObjects_EcalWeightXtalGroups_H

#include <map>
#include <boost/cstdint.hpp>
#include "CondFormats/EcalObjects/interface/EcalXtalGroupId.h"


typedef std::map<uint32_t, EcalXtalGroupId> EcalXtalGroupsMap;

class EcalWeightXtalGroups {
  public:
    EcalWeightXtalGroups();
    ~EcalWeightXtalGroups();
    void  setValue(const uint32_t& xtal, const EcalXtalGroupId& group);
    const EcalXtalGroupsMap& getMap() const { return map_; }

  private:
    EcalXtalGroupsMap map_;
};
#endif
