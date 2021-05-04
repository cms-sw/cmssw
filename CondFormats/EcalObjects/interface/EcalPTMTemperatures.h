#ifndef CondFormats_EcalObjects_EcalPTMTemperatures_H
#define CondFormats_EcalObjects_EcalPTMTemperatures_H
/**
 * Author: Paolo Meridiani
 * Created: 14 November 2006
 * $Id: $
 **/

#include "CondFormats/Serialization/interface/Serializable.h"

#include <map>
#include <cstdint>

class EcalPTMTemperatures {
public:
  typedef std::map<uint32_t, float> EcalPTMTemperatureMap;

  EcalPTMTemperatures();
  ~EcalPTMTemperatures();
  void setValue(const uint32_t& id, const float& value);
  const EcalPTMTemperatureMap& getMap() const { return map_; }

private:
  EcalPTMTemperatureMap map_;

  COND_SERIALIZABLE;
};
#endif
