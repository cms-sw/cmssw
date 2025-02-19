#ifndef HcalPedestalWidths_h
#define HcalPedestalWidths_h

/** 
\class HcalPedestalWidths
\author Radek Ofierzynski
POOL container to store PedestalWidth values 4xCapId, using template
*/

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidth.h"

//typedef HcalCondObjectContainer<HcalPedestalWidth> HcalPedestalWidths;

class HcalPedestalWidths: public HcalCondObjectContainer<HcalPedestalWidth>
{
 public:
  //constructor definition: has to contain 
  HcalPedestalWidths():HcalCondObjectContainer<HcalPedestalWidth>(), unitIsADC(false) {}
  HcalPedestalWidths(bool isADC):HcalCondObjectContainer<HcalPedestalWidth>(), unitIsADC(isADC) {}

  // are the units ADC ? (true=ADC, false=fC)
  bool isADC() const {return unitIsADC;}
  // set unit boolean
  void setUnitADC(bool isADC) {unitIsADC = isADC;}

  std::string const myname() {return (std::string)"HcalPedestalWidths";}

 private:
  bool unitIsADC;

};

#endif
