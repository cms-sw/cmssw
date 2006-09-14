#ifndef CALIBCALORIMETRY_HCALTPGALGOS_HCALNOMINALTPGCODER_H
#define CALIBCALORIMETRY_HCALTPGALGOS_HCALNOMINALTPGCODER_H 1

#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"
#include "CalibFormats/HcalObjects/interface/HcalNominalCoder.h"
#include <vector>

/** \class HcaluLUTTPGCoder
  *  
  * The nominal coder uses a user-supplied table to linearize the ADC values.
  *
  * $Date: 2006/08/31 20:57:06 $
  * $Revision: 1.1 $
  * \author M. Weinberger -- TAMU
  */
class HcaluLUTTPGCoder : public HcalTPGCoder {
public:
  HcaluLUTTPGCoder(const char* filename);
  virtual ~HcaluLUTTPGCoder() {}
  virtual void adc2Linear(const HBHEDataFrame& df, IntegerCaloSamples& ics) const ;
  virtual void adc2Linear(const HFDataFrame& df, IntegerCaloSamples& ics) const;
private:
  void loadLUTs(const char* filename);
  typedef std::vector<int> InputLUT;
  std::vector<InputLUT> luts_;
  const InputLUT* ietaLutMap_[41];
};

#endif
