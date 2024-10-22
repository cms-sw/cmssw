#ifndef EcalTPGLutIdMap_h
#define EcalTPGLutIdMap_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <map>
#include "CondFormats/EcalObjects/interface/EcalTPGLut.h"
#include <cstdint>

class EcalTPGLutIdMap {
public:
  typedef std::map<uint32_t, EcalTPGLut> EcalTPGLutMap;
  typedef std::map<uint32_t, EcalTPGLut>::const_iterator EcalTPGLutMapItr;

  EcalTPGLutIdMap();
  ~EcalTPGLutIdMap();

  const EcalTPGLutMap& getMap() const { return map_; }
  void setValue(const uint32_t& id, const EcalTPGLut& value);

private:
  EcalTPGLutMap map_;

  COND_SERIALIZABLE;
};

#endif
