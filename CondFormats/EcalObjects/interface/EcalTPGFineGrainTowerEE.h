#ifndef EcalTPGFineGrainTowerEE_h
#define EcalTPGFineGrainTowerEE_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <map>
#include <cstdint>

class EcalTPGFineGrainTowerEE {
public:
  EcalTPGFineGrainTowerEE();
  ~EcalTPGFineGrainTowerEE();

  // map<stripId, lut>
  const std::map<uint32_t, uint32_t>& getMap() const { return map_; }
  void setValue(const uint32_t& id, const uint32_t& lut);

private:
  std::map<uint32_t, uint32_t> map_;

  COND_SERIALIZABLE;
};

typedef std::map<uint32_t, uint32_t> EcalTPGFineGrainTowerEEMap;
typedef std::map<uint32_t, uint32_t>::const_iterator EcalTPGFineGrainTowerEEMapIterator;

#endif
