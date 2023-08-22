#ifndef EcalEBPhase2TPGTimeWeightIdMap_h
#define EcalEBPhase2TPGTimeWeightIdMap_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <map>
#include "CondFormats/EcalObjects/interface/EcalEBPhase2TPGTimeWeights.h"
#include <cstdint>

class EcalEBPhase2TPGTimeWeightIdMap {
public:
  typedef std::map<uint32_t, EcalEBPhase2TPGTimeWeights> EcalEBPhase2TPGTimeWeightMap;
  typedef std::map<uint32_t, EcalEBPhase2TPGTimeWeights>::const_iterator EcalEBPhase2TPGTimeWeightMapItr;

  EcalEBPhase2TPGTimeWeightIdMap();
  ~EcalEBPhase2TPGTimeWeightIdMap();

  const EcalEBPhase2TPGTimeWeightMap& getMap() const { return map_; }
  void setValue(const uint32_t& id, const EcalEBPhase2TPGTimeWeights& value);

private:
  EcalEBPhase2TPGTimeWeightMap map_;

  COND_SERIALIZABLE;
};

#endif
