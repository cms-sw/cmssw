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
  * $Date: 2012/11/12 20:43:41 $
  * $Revision: 1.23 $
  * \author M. Weinberger -- TAMU
  * \author Tulika Bose and Greg Landsberg -- Brown
  */
class HcaluLUTTPGCoder : public HcalTPGCoder {
public:

  HcaluLUTTPGCoder();
  virtual ~HcaluLUTTPGCoder();
  virtual void adc2Linear(const HBHEDataFrame& df, IntegerCaloSamples& ics) const;
  virtual void adc2Linear(const HFDataFrame& df, IntegerCaloSamples& ics) const;
  virtual void compress(const IntegerCaloSamples& ics, const std::vector<bool>& featureBits, HcalTriggerPrimitiveDigi& tp) const;
  virtual unsigned short adc2Linear(HcalQIESample sample,HcalDetId id) const;
  virtual float getLUTPedestal(HcalDetId id) const;
  virtual float getLUTGain(HcalDetId id) const;

  void update(const HcalDbService& conditions);
  void update(const char* filename, const HcalTopology&, bool appendMSB = false);
  void updateXML(const char* filename, const HcalTopology&);
  void setLUTGenerationMode(bool gen){ LUTGenerationMode_ = gen; };
  void setMaskBit(int bit){ bitToMask_ = bit; };
  std::vector<unsigned short> getLinearizationLUTWithMSB(const HcalDetId& id) const;
  void lookupMSB(const HBHEDataFrame& df, std::vector<bool>& msb) const;
  bool getMSB(const HcalDetId& id, int adc) const;
  int getLUTId(HcalSubdetector id, int ieta, int iphi, int depth) const;
  int getLUTId(uint32_t rawid) const;
  int getLUTId(const HcalDetId& detid) const;

private:
  // typedef
  typedef unsigned short LutElement;
  typedef std::vector<LutElement> Lut;

  // constants
  static const size_t nluts = 46007, INPUT_LUT_SIZE = 128;
  static const float lsb_;
  
  // member variables
  bool LUTGenerationMode_;
  int bitToMask_;
  std::vector< Lut > inputLUT_;
  std::vector<float> gain_;
  std::vector<float> ped_;
};

#endif
