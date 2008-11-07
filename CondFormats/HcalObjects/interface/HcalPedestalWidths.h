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
  HcalPedestalWidths():HcalCondObjectContainer<HcalPedestalWidth>() {}

  // are the units ADC ? (true=ADC, false=fC)
  const bool isADC() {return unitIsADC;}
  // set unit boolean
  void setUnitADC(bool isADC) {unitIsADC = isADC;}

 private:
  bool unitIsADC;

}

#endif
