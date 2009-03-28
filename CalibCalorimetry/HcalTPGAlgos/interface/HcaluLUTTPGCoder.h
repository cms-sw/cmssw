#ifndef CALIBCALORIMETRY_HCALTPGALGOS_HCALNOMINALTPGCODER_H
#define CALIBCALORIMETRY_HCALTPGALGOS_HCALNOMINALTPGCODER_H 1

#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"
#include "CalibFormats/HcalObjects/interface/HcalNominalCoder.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include <vector>

class HcalDbService;

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
  * $Date: 2008/05/01 20:43:29 $
  * $Revision: 1.15 $
  * \author M. Weinberger -- TAMU
  * \author Tulika Bose and Greg Landsberg -- Brown
  */
class HcaluLUTTPGCoder : public HcalTPGCoder {
public:
 
  HcaluLUTTPGCoder(const char* ifilename, bool read_Ascii_LUTs=false, bool read_XML_LUTs=false);
  virtual ~HcaluLUTTPGCoder();
  virtual void adc2Linear(const HBHEDataFrame& df, IntegerCaloSamples& ics) const;
  virtual void adc2Linear(const HFDataFrame& df, IntegerCaloSamples& ics) const;
  virtual void compress(const IntegerCaloSamples& ics, const std::vector<bool>& featureBits, HcalTriggerPrimitiveDigi& tp) const;
  virtual unsigned short adc2Linear(HcalQIESample sample,HcalDetId id) const;
  virtual float getLUTPedestal(HcalDetId id) const;               // returns the PED for channel id
  virtual float getLUTGain(HcalDetId id) const;              // returns the gain for channel id

  void update(const HcalDbService& conditions);
  void update(const char* filename);
  void updateXML(const char* filename);
  void PrintTPGMap();
  void SetLUTGenerationMode(bool b){ LUTGenerationMode = b; };
  void SetLUTInfo(const std::string& tag, const std::string& algo){
	  TagName = tag;
	  AlgoName = algo;
  };
private:
  void loadILUTs(const char* filename);
  typedef std::vector<int> LUTType;
  std::vector<LUTType> inputluts_;
  static const int nluts = 46007, INPUT_LUT_SIZE = 128;
  int GetLUTID(HcalSubdetector id, int ieta, int iphi, int depth) const;
  int GetLUTID(uint32_t rawid) const;
  void AllocateLUTs();
  void getRecHitCalib(const char* filename);
  float Rcalib[87];
  typedef short unsigned int LUT;
  LUT *inputLUT[nluts];
  float *_gain;
  float *_ped;
  static const float nominal_gain;              // Nominal HB/HE gain in GeV/fC
  bool LUTGenerationMode;
  std::string TagName;
  std::string AlgoName;
};

#endif
