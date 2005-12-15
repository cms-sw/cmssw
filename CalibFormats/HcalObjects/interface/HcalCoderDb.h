#ifndef HCAL_CODER_DB_H
#define HCAL_CODER_DB_H

#include "CalibFormats/HcalObjects/interface/HcalChannelCoder.h"
#include "CalibFormats/HcalObjects/interface/QieShape.h"
#include "CalibFormats/HcalObjects/interface/HcalCoder.h"

/** \class HcalNominalCoder
    
    coder which uses DB services to convert to fC
    $Author: ratnikov
    $Date: 2005/10/05 00:37:56 $
    $Revision: 1.2 $
*/

class HcalQIECoder;
class HcalQIEShape;

class HcalCoderDb : public HcalCoder {
public:
  HcalCoderDb (const HcalQIECoder& fCoder, const HcalQIEShape& fShape);

  virtual void adc2fC(const HBHEDataFrame& df, CaloSamples& lf) const;
  virtual void adc2fC(const HODataFrame& df, CaloSamples& lf) const;
  virtual void adc2fC(const HFDataFrame& df, CaloSamples& lf) const;
  virtual void fC2adc(const CaloSamples& clf, HBHEDataFrame& df) const;
  virtual void fC2adc(const CaloSamples& clf, HFDataFrame& df) const;
  virtual void fC2adc(const CaloSamples& clf, HODataFrame& df) const;

 private:
  template <class Digi> void adc2fC_ (const Digi& df, CaloSamples& clf) const;
  template <class Digi> void fC2adc_ (const CaloSamples& clf, Digi& df, int fCapIdOffset = 0) const;

  const HcalQIECoder* mCoder;
  const HcalQIEShape* mShape;
};

#endif
