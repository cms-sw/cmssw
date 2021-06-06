#ifndef CondFormats_EcalObject_EcalLiteDTUPedestals_h
#define CondFormats_EcalObject_EcalLiteDTUPedestals_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"
#include "DataFormats/EcalDigi/interface/EcalConstants.h"

class EcalLiteDTUPedestals {
public:
  int setMean(unsigned int i, float value) {
    if (i >= ecalPh2::NGAINS)
      return -1;
    else
      meanarray[i] = value;
    return 1;
  }

  int setRMS(unsigned int i, float value) {
    if (i >= ecalPh2::NGAINS)
      return -1;
    else
      rmsarray[i] = value;
    return 1;
  }

  float mean(unsigned int i) const { return meanarray[i]; }

  float rms(unsigned int i) const { return rmsarray[i]; }

private:
  float meanarray[ecalPh2::NGAINS] = {13., 8.};
  float rmsarray[ecalPh2::NGAINS] = {2.8, 1.2};
  COND_SERIALIZABLE;
};

typedef EcalCondObjectContainer<EcalLiteDTUPedestals> EcalLiteDTUPedestalsMap;
typedef EcalLiteDTUPedestalsMap::const_iterator EcalLiteDTUPedestalsMapIterator;

#endif
