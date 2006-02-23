#ifndef CondFormats_EcalObjects_EcalGainRatios_H
#define CondFormats_EcalObjects_EcalGainRatios_H
/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Created: 22 Feb 2006
 * $Id: $
 **/


#include <map>
#include <boost/cstdint.hpp>
#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"


class EcalGainRatios {
  public:
    typedef std::map<uint32_t, EcalMGPAGainRatio> EcalGainRatioMap;

    EcalGainRatios();
    ~EcalGainRatios();
    void  setValue(const uint32_t& id, const EcalMGPAGainRatio& value);
    const EcalGainRatioMap& getMap() const { return map_; }

  private:
    EcalGainRatioMap map_;
};
#endif
