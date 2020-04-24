#ifndef EcalPulseCovariances_h
#define EcalPulseCovariances_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/EcalObjects/interface/EcalPulseShapes.h"
#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"

struct EcalPulseCovariance {

public:

  EcalPulseCovariance();
  
  float covval[EcalPulseShape::TEMPLATESAMPLES][EcalPulseShape::TEMPLATESAMPLES];
  
  float val(int i, int j) const { return covval[i][j]; }

  COND_SERIALIZABLE;

};

typedef EcalCondObjectContainer<EcalPulseCovariance> EcalPulseCovariancesMap;
typedef EcalPulseCovariancesMap::const_iterator EcalPulseCovariancesMapIterator;
typedef EcalPulseCovariancesMap EcalPulseCovariances;

#endif
