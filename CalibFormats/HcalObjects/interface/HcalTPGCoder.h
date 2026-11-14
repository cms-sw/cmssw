#ifndef CalibFormats_HcalObjects_HcalTPGCoder_h
#define CalibFormats_HcalObjects_HcalTPGCoder_h

/** \class HcalTPGCoder
  *  
  * Converts ADC to linear E or ET for use in the TPG path
  * Also compresses linear scale for transmission to RCT
  * 
  * Note : whether the coder produces E or ET is determined by the specific
  * implementation of the coder.
  *
  * \author J. Mans - Minnesota
  */

#include "CalibFormats/CaloObjects/interface/IntegerCaloSamples.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"
#include "DataFormats/HcalDigi/interface/QIE10DataFrame.h"
#include "DataFormats/HcalDigi/interface/QIE11DataFrame.h"

#include <vector>

class HcalTPGCoder {
public:
  virtual ~HcalTPGCoder() = default;
  virtual void adc2Linear(const HBHEDataFrame& df, IntegerCaloSamples& ics) const = 0;
  virtual void adc2Linear(const HFDataFrame& df, IntegerCaloSamples& ics) const = 0;
  virtual void adc2Linear(const QIE10DataFrame& df, IntegerCaloSamples& ics, bool ootpu_lut) const = 0;
  virtual void adc2Linear(const QIE11DataFrame& df, IntegerCaloSamples& ics) const = 0;
  virtual unsigned short adc2Linear(const HcalQIESample& sample, const HcalDetId& id) const = 0;
  unsigned short adc2Linear(unsigned char adc, const HcalDetId& id) const {
    return adc2Linear(HcalQIESample(adc, 0, 0, 0), id);
  }
  virtual void compress(const IntegerCaloSamples& ics,
                        const std::vector<bool>& featureBits,
                        HcalTriggerPrimitiveDigi& tp) const = 0;
  /** \brief Get the full linearization LUT (128 elements).
      Default implementation just uses adc2Linear to get all values
  */
  virtual std::vector<unsigned short> getLinearizationLUT(const HcalDetId& id) const;
  virtual std::vector<unsigned short> getLinearizationLUT(const HcalZDCDetId& id, bool ootput_lut) const;
};

#endif
