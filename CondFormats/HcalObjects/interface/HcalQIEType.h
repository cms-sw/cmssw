#ifndef HcalQIEType_h
#define HcalQIEType_h

/*
\class HcalQIEType
\author Walter Alda
POOL object to store Hcal QIEType
*/

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalQIETypes.h"

//typedef HcalCondObjectContainer<HcalQIETypes> HcalQIEType;

class HcalQIEType: public HcalCondObjectContainer<HcalQIETypes>
{
 public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  HcalQIEType():HcalCondObjectContainer<HcalQIETypes>(0) {}
#endif
  HcalQIEType(const HcalTopology* topo):HcalCondObjectContainer<HcalQIETypes>(topo) {}

  std::string myname() const {return (std::string)"HcalQIEType";}

 private:

 COND_SERIALIZABLE;
};

#endif
