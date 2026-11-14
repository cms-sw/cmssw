#ifndef CalibCalorimetry_CaloTPG_CaloTPGTranscoderULUT_h
#define CalibCalorimetry_CaloTPG_CaloTPGTranscoderULUT_h

/** \class CaloTPGTranscoderULUT
  *  
  * \author J. Mans - Minnesota
  */

#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CondFormats/HcalObjects/interface/HcalLutMetadata.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <set>
#include <vector>

class CaloTPGTranscoderULUT : public CaloTPGTranscoder {
public:
  CaloTPGTranscoderULUT();
  ~CaloTPGTranscoderULUT() override;
  HcalTriggerPrimitiveSample hcalCompress(const HcalTrigTowerDetId& id,
                                          unsigned int sample,
                                          int fineGrain) const override;
  EcalTriggerPrimitiveSample ecalCompress(const EcalTrigTowerDetId& id,
                                          unsigned int sample,
                                          bool fineGrain) const override;

  double hcaletValue(const int& ieta, const int& iphi, const int& version, const int& compressedValue) const override;
  double hcaletValue(const HcalTrigTowerDetId& hid, const HcalTriggerPrimitiveSample& hc) const override;
  virtual bool HTvalid(const int ieta, const int iphi, const int version) const;
  virtual const std::vector<unsigned int> getCompressionLUT(const HcalTrigTowerDetId& id) const;
  virtual void setup(HcalLutMetadata const&,
                     HcalTrigTowerGeometry const&,
                     int nctScaleShift,
                     int rctScaleShift,
                     double lsbQIE8,
                     double lsbQIE11,
                     bool allLinear);
  virtual int getOutputLUTId(const HcalTrigTowerDetId& id) const;
  virtual int getOutputLUTId(const int ieta, const int iphi, const int version) const;

private:
  // Constant
  static const int NOUTLUTS = 4176;

  // Two possible linearization scales
  static const unsigned int REDUCE10BIT = 1024;
  static const unsigned int REDUCE11BIT = 2048;

  // Map different QIE to the right linearization
  static const unsigned int QIE8_OUTPUT_LUT_SIZE = REDUCE10BIT;
  static const unsigned int QIE10_OUTPUT_LUT_SIZE = REDUCE11BIT;
  static const unsigned int QIE11_OUTPUT_LUT_SIZE = REDUCE11BIT;
  static const unsigned int OUTPUT_LUT_SIZE =
      std::max({QIE8_OUTPUT_LUT_SIZE, QIE10_OUTPUT_LUT_SIZE, QIE11_OUTPUT_LUT_SIZE});
  static const unsigned int TPGMAX = 256;

  typedef uint8_t LUT;
  typedef std::array<float, TPGMAX> RCTdecompression;

  const HcalTopology* theTopology;
  static const bool newHFphi = true;

  unsigned int getOutputLUTSize(const HcalTrigTowerDetId& id) const;
  bool isOnlyQIE11(const HcalTrigTowerDetId& id) const;
  void loadHCALCompress(HcalLutMetadata const&, HcalTrigTowerGeometry const&);  //Analytical compression tables

  bool allLinear_ = false;
  double nominal_gain_;
  double lsb_factor_;
  double rct_factor_;
  double nct_factor_;
  double lin8_factor_;
  double lin11_factor_;

  std::vector<std::vector<LUT>> outputLUT_;
  std::vector<RCTdecompression> hcaluncomp_;

  std::set<HcalDetId> plan1_towers_;
};

#endif
