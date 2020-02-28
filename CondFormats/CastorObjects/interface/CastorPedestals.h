#ifndef CastorPedestals_h
#define CastorPedestals_h

/** 
\class CastorPedestals
\author Panos Katsas (UoA)
Modified by L.Mundim (Mar/2009)
POOL container to store Pedestal values 4xCapId
$Author: katsas
*/
#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/CastorObjects/interface/CastorPedestal.h"
#include "CondFormats/CastorObjects/interface/CastorCondObjectContainer.h"

//typedef CastorCondObjectContainer<CastorPedestal> CastorPedestals;

class CastorPedestals : public CastorCondObjectContainer<CastorPedestal> {
public:
  //constructor definition: has to contain
  CastorPedestals() : CastorCondObjectContainer<CastorPedestal>(), unitIsADC(false) {}
  CastorPedestals(bool isADC) : CastorCondObjectContainer<CastorPedestal>(), unitIsADC(isADC) {}

  // are the units ADC ? (true=ADC, false=fC)
  bool isADC() const { return unitIsADC; }
  // set unit boolean
  void setUnitADC(bool isADC) { unitIsADC = isADC; }

private:
  bool unitIsADC;

  COND_SERIALIZABLE;
};

#endif
