#ifndef EcalTPGStripStatus_h
#define EcalTPGStripStatus_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <map>
#include <cstdint>

class EcalTPGStripStatus {
public:
  EcalTPGStripStatus();
  ~EcalTPGStripStatus();

  // map<stripId, status>
  const std::map<uint32_t, uint16_t>& getMap() const { return map_; }
  void setValue(const uint32_t& id, const uint16_t& val);

private:
  std::map<uint32_t, uint16_t> map_;

  COND_SERIALIZABLE;
};

typedef std::map<uint32_t, uint16_t> EcalTPGStripStatusMap;
typedef std::map<uint32_t, uint16_t>::const_iterator EcalTPGStripStatusMapIterator;

#endif
