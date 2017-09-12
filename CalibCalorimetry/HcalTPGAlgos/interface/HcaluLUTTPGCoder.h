#ifndef CALIBCALORIMETRY_HCALTPGALGOS_HCALNOMINALTPGCODER_H
#define CALIBCALORIMETRY_HCALTPGALGOS_HCALNOMINALTPGCODER_H 1

#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"
#include "CalibFormats/HcalObjects/interface/HcalNominalCoder.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include <bitset>
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
  * \author M. Weinberger -- TAMU
  * \author Tulika Bose and Greg Landsberg -- Brown
  */
class HcaluLUTTPGCoder : public HcalTPGCoder {
public:
  static const float  lsb_;

  HcaluLUTTPGCoder(const HcalTopology* topo);
  virtual ~HcaluLUTTPGCoder();
  virtual void adc2Linear(const HBHEDataFrame& df, IntegerCaloSamples& ics) const override;
  virtual void adc2Linear(const HFDataFrame& df, IntegerCaloSamples& ics) const override;
  virtual void adc2Linear(const QIE10DataFrame& df, IntegerCaloSamples& ics) const override;
  virtual void adc2Linear(const QIE11DataFrame& df, IntegerCaloSamples& ics) const override;
  virtual void compress(const IntegerCaloSamples& ics, const std::vector<bool>& featureBits, HcalTriggerPrimitiveDigi& tp) const override;
  virtual unsigned short adc2Linear(HcalQIESample sample,HcalDetId id) const override;
  virtual float getLUTPedestal(HcalDetId id) const override;
  virtual float getLUTGain(HcalDetId id) const override;
  virtual std::vector<unsigned short> getLinearizationLUT(HcalDetId id) const override;

  void update(const HcalDbService& conditions);
  void update(const char* filename, bool appendMSB = false);
  void updateXML(const char* filename);
  void setLUTGenerationMode(bool gen){ LUTGenerationMode_ = gen; };
  void setMaskBit(int bit){ bitToMask_ = bit; };
  std::vector<unsigned short> getLinearizationLUTWithMSB(const HcalDetId& id) const;
  void lookupMSB(const HBHEDataFrame& df, std::vector<bool>& msb) const;
  void lookupMSB(const QIE11DataFrame& df, std::vector<std::bitset<2>>& msb) const;
  bool getMSB(const HcalDetId& id, int adc) const;
  int getLUTId(HcalSubdetector id, int ieta, int iphi, int depth) const;
  int getLUTId(uint32_t rawid) const;
  int getLUTId(const HcalDetId& detid) const;

  static const int QIE8_LUT_BITMASK = 0x3FF;
  static const int QIE10_LUT_BITMASK = 0x7FF;
  static const int QIE11_LUT_BITMASK = 0x3FF;

private:
  // typedef
  typedef unsigned short LutElement;
  typedef std::vector<LutElement> Lut;

  // constants
  static const size_t INPUT_LUT_SIZE = 128;
  static const size_t UPGRADE_LUT_SIZE = 256;
  static const int    nFi_ = 72;

  static const int QIE8_LUT_MSB = 0x400;
  static const int QIE11_LUT_MSB0 = 0x400;
  static const int QIE11_LUT_MSB1 = 0x800;
  
  // member variables
  const HcalTopology* topo_;
  bool LUTGenerationMode_;
  int  bitToMask_;
  int  firstHBEta_, lastHBEta_, nHBEta_, maxDepthHB_, sizeHB_;
  int  firstHEEta_, lastHEEta_, nHEEta_, maxDepthHE_, sizeHE_;
  int  firstHFEta_, lastHFEta_, nHFEta_, maxDepthHF_, sizeHF_;
  std::vector< Lut > inputLUT_;
  std::vector< Lut > upgradeQIE10LUT_;
  std::vector< Lut > upgradeQIE11LUT_;
  std::vector<int> QIEType_;
  std::vector<float> gain_;
  std::vector<float> ped_;
};

#endif
