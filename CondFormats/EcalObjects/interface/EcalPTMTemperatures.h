#ifndef CondFormats_EcalObjects_EcalPTMTemperatures_H
#define CondFormats_EcalObjects_EcalPTMTemperatures_H
/**
 * Author: Paolo Meridiani
 * Created: 14 November 2006
 * $Id: $
 **/


#include <map>
#include <boost/cstdint.hpp>


class EcalPTMTemperatures {
 public:
  typedef std::map<uint32_t, float> EcalPTMTemperatureMap;
  
  EcalPTMTemperatures();
  ~EcalPTMTemperatures();
  void  setValue(const uint32_t& id, const float& value);
  const EcalPTMTemperatureMap& getMap() const { return map_; }

 private:
  EcalPTMTemperatureMap map_;
};
#endif
