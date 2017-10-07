#ifndef HcalQIETypes_h
#define HcalQIETypes_h

/*
 * \class HcalQIETypes
 * \author Walter Alda
 * POOL object to store Hcal QIEType
 * */

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalQIEType.h"

//typedef HcalCondObjectContainer<HcalQIEType> HcalQIETypes;

class HcalQIETypes: public HcalCondObjectContainer<HcalQIEType>
{
 public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  HcalQIETypes():HcalCondObjectContainer<HcalQIEType>(nullptr) {}
#endif
  HcalQIETypes(const HcalTopology* topo):HcalCondObjectContainer<HcalQIEType>(topo) {}

  std::string myname() const override {return (std::string)"HcalQIETypes";}

 private:

 COND_SERIALIZABLE;
};

#endif
