#ifndef EcalPulseSymmCovariances_h
#define EcalPulseSymmCovariances_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/EcalObjects/interface/EcalPulseShapes.h"
#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"

struct EcalPulseSymmCovariance {

public:

  EcalPulseSymmCovariance();
  
  float covval[EcalPulseShape::TEMPLATESAMPLES*(EcalPulseShape::TEMPLATESAMPLES+1)/2];
  
  float val(int i, int j) const { 

    int k=-1;
    if(j >= i) k = j + (EcalPulseShape::TEMPLATESAMPLES-1)*i;
    else k = i + (EcalPulseShape::TEMPLATESAMPLES-1)*j;
    return covval[k]; 

  }

  COND_SERIALIZABLE;

};

typedef EcalCondObjectContainer<EcalPulseSymmCovariance> EcalPulseSymmCovariancesMap;
typedef EcalPulseSymmCovariancesMap::const_iterator EcalPulseSymmCovariancesMapIterator;
typedef EcalPulseSymmCovariancesMap EcalPulseSymmCovariances;

#endif
