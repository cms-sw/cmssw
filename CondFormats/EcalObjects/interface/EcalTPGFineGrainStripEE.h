#ifndef EcalTPGFineGrainStripEE_h
#define EcalTPGFineGrainStripEE_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <map>
#include <cstdint>

class EcalTPGFineGrainStripEE {
public:
  EcalTPGFineGrainStripEE();
  ~EcalTPGFineGrainStripEE();

  struct Item {
    uint32_t threshold;
    uint32_t lut;

    COND_SERIALIZABLE;
  };

  const std::map<uint32_t, Item>& getMap() const { return map_; }
  void setValue(const uint32_t& id, const Item& value);

private:
  std::map<uint32_t, Item> map_;

  COND_SERIALIZABLE;
};

typedef std::map<uint32_t, EcalTPGFineGrainStripEE::Item> EcalTPGFineGrainStripEEMap;
typedef std::map<uint32_t, EcalTPGFineGrainStripEE::Item>::const_iterator EcalTPGFineGrainStripEEMapIterator;

#endif
