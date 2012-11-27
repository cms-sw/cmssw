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
  HcalPedestals():HcalCondObjectContainer<HcalPedestal>(), unitIsADC(false) {}
  HcalPedestals(bool isADC):HcalCondObjectContainer<HcalPedestal>(), unitIsADC(isADC) {}

  // are the units ADC ? (true=ADC, false=fC)
  bool isADC() const {return unitIsADC;} 
  // set unit boolean
  void setUnitADC(bool isADC) {unitIsADC = isADC;}

  std::string myname() const {return (std::string)"HcalPedestals";}

 private:
  bool unitIsADC;

};

#endif
