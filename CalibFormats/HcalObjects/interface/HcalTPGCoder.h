#ifndef CALIBFORMATS_HCALOBJECTS_HCALTPGCODER_H
#define CALIBFORMATS_HCALOBJECTS_HCALTPGCODER_H 1

#include "CalibFormats/CaloObjects/interface/IntegerCaloSamples.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"

/** \class HcalTPGCoder
  *  
  * Converts ADC to ET for use in the TPG path
  *
  * $Date: 2006/03/27 21:05:12 $
  * $Revision: 1.1 $
  * \author J. Mans - Minnesota
  */
class HcalTPGCoder {
public:
  virtual void adc2ET(const HBHEDataFrame& df, IntegerCaloSamples& ics) const = 0;
  virtual void adc2ET(const HFDataFrame& df, IntegerCaloSamples& ics) const = 0;
};

#endif
