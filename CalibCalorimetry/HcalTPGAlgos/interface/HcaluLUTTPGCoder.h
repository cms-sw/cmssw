#ifndef CALIBCALORIMETRY_HCALTPGALGOS_HCALNOMINALTPGCODER_H
#define CALIBCALORIMETRY_HCALTPGALGOS_HCALNOMINALTPGCODER_H 1

#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"
#include "CalibFormats/HcalObjects/interface/HcalNominalCoder.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseContainmentManager.h"

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
  static const float lsb_;

  HcaluLUTTPGCoder();
  HcaluLUTTPGCoder(const HcalTopology* topo, const HcalTimeSlew* delay);
  ~HcaluLUTTPGCoder() override;

  void init(const HcalTopology* top, const HcalTimeSlew* delay);

  void adc2Linear(const HBHEDataFrame& df, IntegerCaloSamples& ics) const override;
  void adc2Linear(const HFDataFrame& df, IntegerCaloSamples& ics) const override;
  void adc2Linear(const QIE10DataFrame& df, IntegerCaloSamples& ics) const override;
  void adc2Linear(const QIE11DataFrame& df, IntegerCaloSamples& ics) const override;
  void compress(const IntegerCaloSamples& ics,
                const std::vector<bool>& featureBits,
                HcalTriggerPrimitiveDigi& tp) const override;
  unsigned short adc2Linear(HcalQIESample sample, HcalDetId id) const override;
  float getLUTPedestal(HcalDetId id) const override;
  float getLUTGain(HcalDetId id) const override;
  std::vector<unsigned short> getLinearizationLUT(HcalDetId id) const override;

  double cosh_ieta(int ieta, int depth, HcalSubdetector subdet);
  void make_cosh_ieta_map(void);
  void update(const HcalDbService& conditions);
  void update(const char* filename, bool appendMSB = false);
  void updateXML(const char* filename);
  void setLUTGenerationMode(bool gen) { LUTGenerationMode_ = gen; };
  void setFGHFthresholds(const std::vector<uint32_t>& fgthresholds) { FG_HF_thresholds_ = fgthresholds; };
  void setMaskBit(int bit) { bitToMask_ = bit; };
  void setAllLinear(bool linear, double lsb8, double lsb11, double lsb11overlap) {
    allLinear_ = linear;
    linearLSB_QIE8_ = lsb8;
    linearLSB_QIE11_ = lsb11;
    linearLSB_QIE11Overlap_ = lsb11overlap;
  };
  void set1TSContainHB(bool contain1TSHB) { contain1TSHB_ = contain1TSHB; }
  void set1TSContainHE(bool contain1TSHE) { contain1TSHE_ = contain1TSHE; }
  void setContainPhaseHB(double containPhaseNSHB) { containPhaseNSHB_ = containPhaseNSHB; }
  void setContainPhaseHE(double containPhaseNSHE) { containPhaseNSHE_ = containPhaseNSHE; }
  void lookupMSB(const HBHEDataFrame& df, std::vector<bool>& msb) const;
  void lookupMSB(const QIE10DataFrame& df, std::vector<std::bitset<2>>& msb) const;
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
  static const int nFi_ = 72;

  static const int QIE8_LUT_MSB = 0x400;
  static const int QIE11_LUT_MSB0 = 0x400;
  static const int QIE11_LUT_MSB1 = 0x800;
  static const int QIE10_LUT_MSB0 = 0x1000;
  static const int QIE10_LUT_MSB1 = 0x2000;

  // member variables
  const HcalTopology* topo_;
  const HcalTimeSlew* delay_;
  bool LUTGenerationMode_;
  std::vector<uint32_t> FG_HF_thresholds_;
  int bitToMask_;
  int firstHBEta_, lastHBEta_, nHBEta_, maxDepthHB_, sizeHB_;
  int firstHEEta_, lastHEEta_, nHEEta_, maxDepthHE_, sizeHE_;
  int firstHFEta_, lastHFEta_, nHFEta_, maxDepthHF_, sizeHF_;
  std::vector<Lut> inputLUT_;
  std::vector<float> gain_;
  std::vector<float> ped_;
  std::vector<double> cosh_ieta_;
  // edge cases not covered by the cosh_ieta_ map
  double cosh_ieta_28_HE_low_depths_, cosh_ieta_28_HE_high_depths_, cosh_ieta_29_HE_;
  bool allLinear_;
  bool contain1TSHB_, contain1TSHE_;
  double containPhaseNSHB_, containPhaseNSHE_;
  double linearLSB_QIE8_, linearLSB_QIE11_, linearLSB_QIE11Overlap_;
  std::unique_ptr<HcalPulseContainmentManager> pulseCorr_;
};

#endif
