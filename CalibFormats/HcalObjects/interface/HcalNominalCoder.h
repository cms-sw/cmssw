#ifndef HCALNOMINALCODER_H
#define HCALNOMINALCODER_H 1

#include "CalibFormats/HcalObjects/interface/HcalCoder.h"

/** \class HcalNominalCoder
    
    Simple coder which uses the QIESample to convert to fC

   $Date: $
   $Revision: $
*/
class HcalNominalCoder : public HcalCoder {
public:
  virtual void adc2fC(const cms::HBHEDataFrame& df, CaloSamples& lf) const;
  virtual void adc2fC(const cms::HODataFrame& df, CaloSamples& lf) const;
  virtual void adc2fC(const cms::HFDataFrame& df, CaloSamples& lf) const;
  virtual void fC2adc(const CaloSamples& clf, cms::HBHEDataFrame& df) const;
  virtual void fC2adc(const CaloSamples& clf, cms::HFDataFrame& df) const;
  virtual void fC2adc(const CaloSamples& clf, cms::HODataFrame& df) const;
};

#endif
