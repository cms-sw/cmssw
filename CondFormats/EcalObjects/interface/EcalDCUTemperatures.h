#ifndef CondFormats_EcalObjects_EcalDCUTemperatures_H
#define CondFormats_EcalObjects_EcalDCUTemperatures_H
/**
 * Author: Paolo Meridiani
 * Created: 14 November 2006
 * $Id: EcalDCUTemperatures.h,v 1.1 2006/11/16 18:19:44 meridian Exp $
 **/


#include <map>
#include <boost/cstdint.hpp>


class EcalDCUTemperatures {
 public:
  typedef std::map<uint32_t, float> EcalDCUTemperatureMap;
  
  EcalDCUTemperatures();
  ~EcalDCUTemperatures();
  void  setValue(const uint32_t& id, const float& value);
  const EcalDCUTemperatureMap& getMap() const { return map_; }

 private:
  EcalDCUTemperatureMap map_;
};
#endif
