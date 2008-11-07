#ifndef HcalPedestals_h
#define HcalPedestals_h

/** 
\class HcalPedestals
\author Radek Ofierzynski
POOL container to store Pedestal values 4xCapId, using template
*/

#include "CondFormats/HcalObjects/interface/HcalPedestal.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"

//typedef HcalCondObjectContainer<HcalPedestal> HcalPedestals;

class HcalPedestals: public HcalCondObjectContainer<HcalPedestal>
{
 public:
  //constructor definition: has to contain 
  HcalPedestals(bool isADC):HcalCondObjectContainer<HcalPedestal>():unitIsADC(isADC) {}

  // are the units ADC ? (true=ADC, false=fC)
  const bool isADC() {return unitIsADC;}
  // set unit boolean
  //  void setUnitADC(bool isADC) {unitIsADC = isADC;}
  // only via constructor

 private:
  bool unitIsADC;

};

#endif
