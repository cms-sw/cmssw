#ifndef HcalChannelQuality_h
#define HcalChannelQuality_h

/** 
\class HcalChannelQuality
\author Radek Ofierzynski
POOL object to store HcalChannelStatus
*/

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"

//typedef HcalCondObjectContainer<HcalChannelStatus> HcalChannelQuality;

class HcalChannelQuality: public HcalCondObjectContainer<HcalChannelStatus>
{
 public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  HcalChannelQuality():HcalCondObjectContainer<HcalChannelStatus>(0) {}
#endif
  HcalChannelQuality(const HcalTopology* topo):HcalCondObjectContainer<HcalChannelStatus>(topo) {}

  std::string myname() const {return (std::string)"HcalChannelQuality";}

 private:
};


#endif

