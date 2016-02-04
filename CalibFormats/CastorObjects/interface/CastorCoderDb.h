#ifndef CASTOR_CODER_DB_H
#define CASTOR_CODER_DB_H

#include "CalibFormats/CastorObjects/interface/CastorChannelCoder.h"
#include "CalibFormats/CastorObjects/interface/QieShape.h"
#include "CalibFormats/CastorObjects/interface/CastorCoder.h"

/** \class CastorCoderDb
    
    coder which uses DB services to convert to fC

*/

class CastorQIECoder;
class CastorQIEShape;

class CastorCoderDb : public CastorCoder {
public:
  CastorCoderDb (const CastorQIECoder& fCoder, const CastorQIEShape& fShape);

  virtual void adc2fC(const CastorDataFrame& df, CaloSamples& lf) const;
 
  virtual void fC2adc(const CaloSamples& clf, CastorDataFrame& df, int fCapIdOffset) const;
 
 private:
  template <class Digi> void adc2fC_ (const Digi& df, CaloSamples& clf) const;
  template <class Digi> void fC2adc_ (const CaloSamples& clf, Digi& df, int fCapIdOffset) const;

  const CastorQIECoder* mCoder;
  const CastorQIEShape* mShape;
};

#endif
