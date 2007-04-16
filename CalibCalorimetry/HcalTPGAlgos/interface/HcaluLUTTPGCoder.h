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
  * $Date: 2007/04/04 04:12:46 $
  * $Revision: 1.6 $
  * \author M. Weinberger -- TAMU
  * \author Tulika Bose and Greg Landsberg -- Brown
  */
class HcaluLUTTPGCoder : public HcalTPGCoder {
public:
  HcaluLUTTPGCoder(const char* filename);
  HcaluLUTTPGCoder(const char* ifilename, const char* ofilename);
  virtual ~HcaluLUTTPGCoder();
  virtual void adc2Linear(const HBHEDataFrame& df, IntegerCaloSamples& ics) const;
  virtual void adc2Linear(const HFDataFrame& df, IntegerCaloSamples& ics) const;
<<<<<<< HcaluLUTTPGCoder.h
  virtual void compress(const IntegerCaloSamples& ics, const std::vector<bool>& featureBits, HcalTriggerPrimitiveDigi& tp) const;
  virtual void getConditions(const edm::EventSetup& es) const;
  virtual void releaseConditions() const {}
  
=======
  virtual void compress(const IntegerCaloSamples& ics, const std::vector<bool>& featureBits, HcalTriggerPrimitiveDigi& tp) const;  
  bool getadc2fCLUT();
  bool getped();
  bool getgain();
>>>>>>> 1.6
private:
<<<<<<< HcaluLUTTPGCoder.h
  static const int nluts = 46005, INPUT_LUT_SIZE = 128;
  int GetLUTID(HcalSubdetector id, int ieta, int iphi, int depth) const;
  void AllocateLUTs();
  void getRecHitCalib(const char* filename);
  float Rcalib[87];
  typedef short unsigned int LUT;
  LUT *inputLUT[nluts];
  static const float nominal_gain = 0.177;              // Nominal HB/HE gain in GeV/fC
=======
  void loadILUTs(const char* filename);
  void loadOLUTs(const char* filename);
  //void generateILUTs(const char *filename);
  void generateILUTs();
  typedef std::vector<int> LUTType;
  std::vector<LUTType> inputluts_;
  const LUTType* ietaILutMap_[54];
  std::vector<LUTType> outputluts_;
  const LUTType* ietaOLutMap_[32];
  float adc2fCLUT_[128];
  //std::vector<LUTType> adc2fCLUT_;
  float ped_;
  float ped_HF;
  float gain_;
>>>>>>> 1.6
};
#endif
