#ifndef EcalEBPhase2TPGAmplWeightIdMap_h
#define EcalEBPhase2TPGAmplWeightIdMap_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <map>
#include "CondFormats/EcalObjects/interface/EcalEBPhase2TPGAmplWeights.h"
#include <cstdint>

class EcalEBPhase2TPGAmplWeightIdMap {
public:
  typedef std::map<uint32_t, EcalEBPhase2TPGAmplWeights> EcalEBPhase2TPGAmplWeightMap;
  typedef std::map<uint32_t, EcalEBPhase2TPGAmplWeights>::const_iterator EcalEBPhase2TPGAmplWeightMapItr;

  EcalEBPhase2TPGAmplWeightIdMap();
  ~EcalEBPhase2TPGAmplWeightIdMap();

  const EcalEBPhase2TPGAmplWeightMap& getMap() const { return map_; }
  void setValue(const uint32_t& id, const EcalEBPhase2TPGAmplWeights& value);

private:
  EcalEBPhase2TPGAmplWeightMap map_;

  COND_SERIALIZABLE;
};

#endif
