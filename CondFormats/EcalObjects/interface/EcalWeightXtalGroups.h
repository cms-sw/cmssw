#ifndef CondFormats_EcalObjects_EcalWeightXtalGroups_H
#define CondFormats_EcalObjects_EcalWeightXtalGroups_H
/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Created: 22 Feb 2006
 * $Id: EcalWeightXtalGroups.h,v 1.2 2006/02/23 16:56:34 rahatlou Exp $
 **/

#include "DataFormats/EcalDetId/interface/EcalContainer.h"
#include <map>
#include <boost/cstdint.hpp>
#include "CondFormats/EcalObjects/interface/EcalXtalGroupId.h"

class EcalWeightXtalGroups {
public:
  typedef  EcalXtalGroupId Item; 
  typedef std::map<uint32_t, EcalXtalGroupId> EcalXtalGroupsMap;
  
  EcalWeightXtalGroups();
  ~EcalWeightXtalGroups();
  void  setValue(const uint32_t& xtal, const EcalXtalGroupId& group);
  const EcalXtalGroupsMap& getMap() const { return map_; }

  void update() const;

  Item const & operator()(DetId id) const {
    return m_hashedCont(id);
  }

  Item const & barrel(size_t hashid) const {
    return m_hashedCont.barrel(hashid);
  }
  Item const & endcap(size_t hashid) const {
    return m_hashedCont.endcap(hashid);
  }

  
private:
  void doUpdate();
  EcalContainer<Item> m_hashedCont;
  
private:
  EcalXtalGroupsMap map_;
};
#endif
