#ifndef CondFormats_EcalObjects_EcalChannelStatus_H
#define CondFormats_EcalObjects_EcalChannelStatus_H
/**
 * Author: Paolo Meridiani
 * Created: 14 November 2006
 * $Id: $
 **/

#include <map>
#include <boost/cstdint.hpp>
#include "CondFormats/EcalObjects/interface/EcalChannelStatusCode.h"

class EcalChannelStatus {
 public:
  typedef std::map<uint32_t, EcalChannelStatusCode> EcalChannelStatusMap;
  
  EcalChannelStatus();
  ~EcalChannelStatus();
  void  setValue(const uint32_t& id, const EcalChannelStatusCode& value);
  const EcalChannelStatusMap& getMap() const { return map_; }
  
 private:
  EcalChannelStatusMap map_;
};
#endif
