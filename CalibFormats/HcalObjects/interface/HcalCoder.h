#ifndef HCALCODER_H
#define HCALCODER_H 1

#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"

/** \class HcalCoder
    
    Abstract interface of a coder/decoder which converts ADC values to
    and from femtocolumbs of collected charge.

   $Date: $
   $Revision: $
*/
class HcalCoder {
public:
  virtual void adc2fC(const HBHEDataFrame& df, CaloSamples& lf) const = 0;
  virtual void adc2fC(const HODataFrame& df, CaloSamples& lf) const = 0;
  virtual void adc2fC(const HFDataFrame& df, CaloSamples& lf) const = 0;
  virtual void fC2adc(const CaloSamples& clf, HBHEDataFrame& df) const = 0;
  virtual void fC2adc(const CaloSamples& clf, HFDataFrame& df) const = 0;
  virtual void fC2adc(const CaloSamples& clf, HODataFrame& df) const = 0;
};

#endif
