#ifndef CASTORNOMINALCODER_H
#define CASTORNOMINALCODER_H 1

#include "CalibFormats/CastorObjects/interface/CastorCoder.h"

/** \class CastorNominalCoder
    
    Simple coder which uses the QIESample to convert to fC

*/
class CastorNominalCoder : public CastorCoder {
public:
  virtual void adc2fC(const CastorDataFrame& df, CaloSamples& lf) const;

  virtual void fC2adc(const CaloSamples& clf, CastorDataFrame& df, int fCapIdOffset) const; 
};

#endif
