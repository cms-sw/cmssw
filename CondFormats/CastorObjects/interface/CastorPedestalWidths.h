#ifndef CastorPedestalWidths_h
#define CastorPedestalWidths_h

/** 
\class CastorPedestalWidths
\author Radek Ofierzynski
Modified by L.Mundim (Mar/2009)
POOL container to store PedestalWidth values 4xCapId, using template
*/

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/CastorObjects/interface/CastorCondObjectContainer.h"
#include "CondFormats/CastorObjects/interface/CastorPedestalWidth.h"

//typedef CastorCondObjectContainer<CastorPedestalWidth> CastorPedestalWidths;

class CastorPedestalWidths : public CastorCondObjectContainer<CastorPedestalWidth> {
public:
  //constructor definition: has to contain
  CastorPedestalWidths() : CastorCondObjectContainer<CastorPedestalWidth>(), unitIsADC(false) {}
  CastorPedestalWidths(bool isADC) : CastorCondObjectContainer<CastorPedestalWidth>(), unitIsADC(isADC) {}

  // are the units ADC ? (true=ADC, false=fC)
  bool isADC() const { return unitIsADC; }
  // set unit boolean
  void setUnitADC(bool isADC) { unitIsADC = isADC; }

  std::string const myname() { return (std::string) "CastorPedestalWidths"; }

private:
  bool unitIsADC;

  COND_SERIALIZABLE;
};

#endif
