#ifndef CALIBFORMATS_CASTOROBJECTS_CASTORTPGCODER_H
#define CALIBFORMATS_CASTOROBJECTS_CASTORTPGCODER_H 1

#include "CalibFormats/CaloObjects/interface/IntegerCaloSamples.h"
#include "DataFormats/HcalDigi/interface/CastorDataFrame.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

// forward declaration of EventSetup is all that is needed here
namespace edm {
  class EventSetup; 
}

/** \class CastorTPGCoder
  *  
  * Converts ADC to linear E or ET for use in the TPG path
  * Also compresses linear scale for transmission to RCT
  * 
  * Note : whether the coder produces E or ET is determined by the specific
  * implementation of the coder.
  *
  * $Date: 2009/03/26 17:55:06 $
  * $Revision: 1.2 $
  * \author J. Mans - Minnesota
  */
class CastorTPGCoder {
public:

  //  virtual void adc2Linear(const CastorDataFrame& df, IntegerCaloSamples& ics) const = 0;

  virtual unsigned short adc2Linear(HcalQIESample sample,HcalDetId id) const = 0;
  unsigned short adc2Linear(unsigned char adc, HcalDetId id) const { return adc2Linear(HcalQIESample(adc,0,0,0),id); }
  virtual float getLUTPedestal(HcalDetId id) const = 0;
  virtual float getLUTGain(HcalDetId id) const = 0;
  /** \brief Get the full linearization LUT (128 elements).
      Default implementation just uses adc2Linear to get all values
  */
  virtual std::vector<unsigned short> getLinearizationLUT(HcalDetId id) const;
};

#endif
