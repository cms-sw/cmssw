#ifndef CASTORCODER_H
#define CASTORCODER_H 1

#include "DataFormats/HcalDigi/interface/CastorDataFrame.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"

/** \class CastorCoder
    
    Abstract interface of a coder/decoder which converts ADC values to
    and from femtocoulombs of collected charge.

*/
class CastorCoder {
public:
  virtual void adc2fC(const CastorDataFrame& df, CaloSamples& lf) const = 0;
  virtual void fC2adc(const CaloSamples& clf, CastorDataFrame& df, int fCapIdOffset) const = 0;
};

#endif
