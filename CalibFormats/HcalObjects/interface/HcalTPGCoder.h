#ifndef CALIBFORMATS_HCALOBJECTS_HCALTPGCODER_H
#define CALIBFORMATS_HCALOBJECTS_HCALTPGCODER_H 1

#include "CalibFormats/CaloObjects/interface/IntegerCaloSamples.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"

// forward declaration of EventSetup is all that is needed here
namespace edm {
  class EventSetup; 
}

/** \class HcalTPGCoder
  *  
  * Converts ADC to linear E or ET for use in the TPG path
  * 
  * Note : whether the coder produces E or ET is determined by the specific
  * implementation of the coder.
  *
  * $Date: 2006/09/14 16:58:37 $
  * $Revision: 1.3 $
  * \author J. Mans - Minnesota
  */
class HcalTPGCoder {
public:
  virtual void getConditions(const edm::EventSetup& es) const { }
  virtual void releaseConditions() const { }
  virtual void adc2Linear(const HBHEDataFrame& df, IntegerCaloSamples& ics) const = 0;
  virtual void adc2Linear(const HFDataFrame& df, IntegerCaloSamples& ics) const = 0;
};

#endif
