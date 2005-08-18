#ifndef HCAL_CODER_DB_H
#define HCAL_CODER_DB_H

#include "CalibFormats/HcalObjects/interface/HcalChannelCoder.h"
#include "CalibFormats/HcalObjects/interface/QieShape.h"
#include "CalibFormats/HcalObjects/interface/HcalCoder.h"

/** \class HcalNominalCoder
    
    coder which uses DB services to convert to fC
    $Author: ratnikov
    $Date: 2005/08/04 18:16:29 $
    $Revision: 1.1 $
*/

class HcalChannelCoder;
class QieShape;

class HcalCoderDb : public HcalCoder {
public:
  HcalCoderDb (const HcalChannelCoder& fCoder, const QieShape& fShape);

  virtual void adc2fC(const cms::HBHEDataFrame& df, CaloSamples& lf) const;
  virtual void adc2fC(const cms::HODataFrame& df, CaloSamples& lf) const;
  virtual void adc2fC(const cms::HFDataFrame& df, CaloSamples& lf) const;
  virtual void fC2adc(const CaloSamples& clf, cms::HBHEDataFrame& df) const;
  virtual void fC2adc(const CaloSamples& clf, cms::HFDataFrame& df) const;
  virtual void fC2adc(const CaloSamples& clf, cms::HODataFrame& df) const;

 private:
  template <class Digi> void adc2fC_ (const Digi& df, CaloSamples& clf) const;
  template <class Digi> void fC2adc_ (const CaloSamples& clf, Digi& df, int fCapIdOffset = 0) const;

  const HcalChannelCoder* mCoder;
  const QieShape* mShape;
};

#endif
