#ifndef HCALNOMINALCODER_H
#define HCALNOMINALCODER_H 1

#include "CalibFormats/HcalObjects/interface/HcalCoder.h"

/** \class HcalNominalCoder
    
    Simple coder which uses the QIESample to convert to fC

   $Date: 2005/08/04 18:16:29 $
   $Revision: 1.1 $
*/
class HcalNominalCoder : public HcalCoder {
public:
  virtual void adc2fC(const HBHEDataFrame& df, CaloSamples& lf) const;
  virtual void adc2fC(const HODataFrame& df, CaloSamples& lf) const;
  virtual void adc2fC(const HFDataFrame& df, CaloSamples& lf) const;
  virtual void fC2adc(const CaloSamples& clf, HBHEDataFrame& df) const;
  virtual void fC2adc(const CaloSamples& clf, HFDataFrame& df) const;
  virtual void fC2adc(const CaloSamples& clf, HODataFrame& df) const;
};

#endif
