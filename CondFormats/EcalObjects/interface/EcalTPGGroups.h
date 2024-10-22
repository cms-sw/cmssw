#ifndef EcalTPGGroups_h
#define EcalTPGGroups_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <map>
#include <cstdint>

/*
this class is used to define groups which associate a rawId to an objectId where:
- rawId (simple integer) refers to a crystal (i.e EBDetId.rawId()), a (pseudo)strip or a tower
- objectId (simple integer) refers to a LUTid, a FineGrainEBId or a WeightId

P.P.
*/

class EcalTPGGroups {
public:
  typedef std::map<uint32_t, uint32_t> EcalTPGGroupsMap;
  typedef std::map<uint32_t, uint32_t>::const_iterator EcalTPGGroupsMapItr;

  EcalTPGGroups();
  ~EcalTPGGroups();

  const EcalTPGGroupsMap& getMap() const { return map_; }
  void setValue(const uint32_t& rawId, const uint32_t& ObjectId);

protected:
  EcalTPGGroupsMap map_;

  COND_SERIALIZABLE;
};

#endif
