#ifndef CALIBFORMATS_HCALOBJECTS_HCALTPGCODER_H
#define CALIBFORMATS_HCALOBJECTS_HCALTPGCODER_H 1

#include "CalibFormats/CaloObjects/interface/IntegerCaloSamples.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"

// forward declaration of EventSetup is all that is needed here
namespace edm {
  class EventSetup; 
}

/** \class HcalTPGCoder
  *  
  * Converts ADC to linear E or ET for use in the TPG path
  * Also compresses linear scale for transmission to RCT
  * 
  * Note : whether the coder produces E or ET is determined by the specific
  * implementation of the coder.
  *
  * $Date: 2007/07/11 22:06:36 $
  * $Revision: 1.6 $
  * \author J. Mans - Minnesota
  */
class HcalTPGCoder {
public:
  virtual void adc2Linear(const HBHEDataFrame& df, IntegerCaloSamples& ics) const = 0;
  virtual void adc2Linear(const HFDataFrame& df, IntegerCaloSamples& ics) const = 0;
  virtual unsigned short adc2Linear(HcalQIESample sample,HcalDetId id) const = 0;
  unsigned short adc2Linear(unsigned char adc, HcalDetId id) const { return adc2Linear(HcalQIESample(adc,0,0,0),id); }
  virtual void compress(const IntegerCaloSamples& ics, const std::vector<bool>& featureBits, HcalTriggerPrimitiveDigi& tp) const = 0;
  virtual float getLUTPedestal(HcalDetId id) const = 0;
  virtual float getLUTGain(HcalDetId id) const = 0;
  /** \brief Get the full linearization LUT (128 elements).
      Default implementation just uses adc2Linear to get all values
  */
  virtual std::vector<unsigned short> getLinearizationLUT(HcalDetId id) const;
};

#endif
