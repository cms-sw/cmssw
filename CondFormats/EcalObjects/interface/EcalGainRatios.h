#ifndef CondFormats_EcalObjects_EcalGainRatios_H
#define CondFormats_EcalObjects_EcalGainRatios_H

#include <map>
#include <boost/cstdint.hpp>
#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"


typedef std::map<uint32_t, EcalMGPAGainRatio> EcalGainRatioMap;

class EcalGainRatios {
  public:
    EcalGainRatios();
    ~EcalGainRatios();
    void  setValue(const uint32_t& id, const EcalMGPAGainRatio& value);
    const EcalGainRatioMap& getMap() const { return map_; }

  private:
    EcalGainRatioMap map_;
};
#endif
