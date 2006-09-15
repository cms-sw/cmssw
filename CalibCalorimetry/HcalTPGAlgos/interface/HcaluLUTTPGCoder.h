#ifndef CALIBCALORIMETRY_HCALTPGALGOS_HCALNOMINALTPGCODER_H
#define CALIBCALORIMETRY_HCALTPGALGOS_HCALNOMINALTPGCODER_H 1

#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"
#include "CalibFormats/HcalObjects/interface/HcalNominalCoder.h"
#include <vector>

/** \class HcaluLUTTPGCoder
  *  
  * The nominal coder uses a user-supplied table to linearize the ADC values.
  *
  * <pre>
  * [number of ieta slices]
  * [low tower 1] [low tower 2] ...
  * [high tower 1] [ high tower 2] ...
  * [LUT 1(0)] [LUT 2(0)] ...
  * [LUT 1(1)] [LUT 2(1)] ...
  * . . .
  * [LUT 1(127)] [LUT 2(127)] ...
  * </pre>
  *
  * $Date: 2006/09/14 19:41:18 $
  * $Revision: 1.2 $
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
