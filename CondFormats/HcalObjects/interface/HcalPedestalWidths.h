#ifndef HcalPedestalWidths_h
#define HcalPedestalWidths_h

/** 
\class HcalPedestalWidths
\author Radek Ofierzynski
POOL container to store PedestalWidth values 4xCapId, using template
*/

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidth.h"

//typedef HcalCondObjectContainer<HcalPedestalWidth> HcalPedestalWidths;

class HcalPedestalWidths: public HcalCondObjectContainer<HcalPedestalWidth>
{
 public:
  //constructor definition: has to contain 
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  HcalPedestalWidths():HcalCondObjectContainer<HcalPedestalWidth>(nullptr), unitIsADC(false) {}
#endif
  HcalPedestalWidths(const HcalTopology* topo):HcalCondObjectContainer<HcalPedestalWidth>(topo), unitIsADC(false) {}
  HcalPedestalWidths(const HcalTopology* topo,bool isADC):HcalCondObjectContainer<HcalPedestalWidth>(topo), unitIsADC(isADC) {}

  // are the units ADC ? (true=ADC, false=fC)
  bool isADC() const {return unitIsADC;}
  // set unit boolean
  void setUnitADC(bool isADC) {unitIsADC = isADC;}

  std::string myname() const override {return (std::string)"HcalPedestalWidths";}

 private:
  bool unitIsADC;


 COND_SERIALIZABLE;
};

#endif
