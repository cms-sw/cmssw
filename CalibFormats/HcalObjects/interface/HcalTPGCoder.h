#ifndef CALIBFORMATS_HCALOBJECTS_HCALTPGCODER_H
#define CALIBFORMATS_HCALOBJECTS_HCALTPGCODER_H 1

#include "CalibFormats/CaloObjects/interface/IntegerCaloSamples.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"

/** \class HcalTPGCoder
  *  
  * Converts ADC to ET for use in the TPG path
  *
  * $Date: $
  * $Revision: $
  * \author J. Mans - Minnesota
  */
class HcalTPGCoder {
public:
  virtual void adc2ET(const HBHEDataFrame& df, IntegerCaloSamples& ics) const = 0;
  virtual void adc2ET(const HODataFrame& df, IntegerCaloSamples& ics) const = 0;
  virtual void adc2ET(const HFDataFrame& df, IntegerCaloSamples& ics) const = 0;
};

#endif
