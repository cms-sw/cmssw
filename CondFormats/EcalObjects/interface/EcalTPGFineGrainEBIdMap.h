#ifndef EcalTPGFineGrainEBIdMap_h
#define EcalTPGFineGrainEBIdMap_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <map>
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainConstEB.h"
#include <cstdint>

class EcalTPGFineGrainEBIdMap {
public:
  typedef std::map<uint32_t, EcalTPGFineGrainConstEB> EcalTPGFineGrainEBMap;
  typedef std::map<uint32_t, EcalTPGFineGrainConstEB>::const_iterator EcalTPGFineGrainEBMapItr;

  EcalTPGFineGrainEBIdMap();
  ~EcalTPGFineGrainEBIdMap();

  const EcalTPGFineGrainEBMap& getMap() const { return map_; }
  void setValue(const uint32_t& id, const EcalTPGFineGrainConstEB& value);

private:
  EcalTPGFineGrainEBMap map_;

  COND_SERIALIZABLE;
};

#endif
