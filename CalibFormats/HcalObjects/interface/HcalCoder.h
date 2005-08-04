#ifndef HCALCODER_H
#define HCALCODER_H 1

#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"

/** \class HcalCoder
    
    Abstract interface of a coder/decoder which converts ADC values to
    and from femtocolumbs of collected charge.

   $Date: 2005/07/27 19:44:31 $
   $Revision: 1.1 $
*/
class HcalCoder {
public:
  virtual void adc2fC(const cms::HBHEDataFrame& df, CaloSamples& lf) const = 0;
  virtual void adc2fC(const cms::HODataFrame& df, CaloSamples& lf) const = 0;
  virtual void adc2fC(const cms::HFDataFrame& df, CaloSamples& lf) const = 0;
  virtual void fC2adc(const CaloSamples& clf, cms::HBHEDataFrame& df) const = 0;
  virtual void fC2adc(const CaloSamples& clf, cms::HFDataFrame& df) const = 0;
  virtual void fC2adc(const CaloSamples& clf, cms::HODataFrame& df) const = 0;
};

#endif
