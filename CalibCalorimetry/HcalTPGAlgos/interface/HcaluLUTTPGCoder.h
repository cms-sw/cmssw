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
  * $Date: 2006/09/15 18:58:48 $
  * $Revision: 1.3 $
  * \author M. Weinberger -- TAMU
  */
class HcaluLUTTPGCoder : public HcalTPGCoder {
public:
  HcaluLUTTPGCoder(const char* filename);
  HcaluLUTTPGCoder(const char* ifilename, const char* ofilename);
  virtual ~HcaluLUTTPGCoder() {}
  virtual void adc2Linear(const HBHEDataFrame& df, IntegerCaloSamples& ics) const ;
  virtual void adc2Linear(const HFDataFrame& df, IntegerCaloSamples& ics) const;
  virtual void compress(const IntegerCaloSamples& ics, const std::vector<bool>& featureBits, HcalTriggerPrimitiveDigi& tp) const;  
private:
  void loadILUTs(const char* filename);
  void loadOLUTs(const char* filename);
  typedef std::vector<int> LUTType;
  std::vector<LUTType> inputluts_;
  const LUTType* ietaILutMap_[41];
  std::vector<LUTType> outputluts_;
  const LUTType* ietaOLutMap_[32];
  
};

#endif
