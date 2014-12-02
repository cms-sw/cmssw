#ifndef CALOTPGTRANSCODERULUT_H
#define CALOTPGTRANSCODERULUT_H 1

#include <memory>
#include <vector>
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"


// tmp
#include "CondFormats/HcalObjects/interface/HcalLutMetadata.h"


/** \class CaloTPGTranscoderULUT
  *
  * \author J. Mans - Minnesota
  */

class HcalTrigTowerGeometry;

class CaloTPGTranscoderULUT : public CaloTPGTranscoder {
public:
  CaloTPGTranscoderULUT(const std::string& compressionFile="",
                        const std::string& decompressionFile="");
  virtual ~CaloTPGTranscoderULUT();
  virtual HcalTriggerPrimitiveSample hcalCompress(const HcalTrigTowerDetId& id, unsigned int sample, bool fineGrain, HcalTrigTowerGeometry const&) const;
  virtual EcalTriggerPrimitiveSample ecalCompress(const EcalTrigTowerDetId& id, unsigned int sample, bool fineGrain) const;

  virtual void rctEGammaUncompress(const HcalTrigTowerDetId& hid, const HcalTriggerPrimitiveSample& hc,
				   const EcalTrigTowerDetId& eid, const EcalTriggerPrimitiveSample& ec,
				   unsigned int& et, bool& egVecto, bool& activity) const;
  virtual void rctJetUncompress(const HcalTrigTowerDetId& hid, const HcalTriggerPrimitiveSample& hc,
				   const EcalTrigTowerDetId& eid, const EcalTriggerPrimitiveSample& ec,
				   unsigned int& et) const;
  virtual double hcaletValue(const HcalTrigTowerDetId& hid, const HcalTriggerPrimitiveSample& hc, HcalTrigTowerGeometry const&) const;
  virtual std::vector<unsigned char> getCompressionLUT(HcalTrigTowerDetId id, HcalTopology const&) const;
  virtual void setup(HcalLutMetadata const&, HcalTrigTowerGeometry const&);
  //virtual int getOutputLUTId(const int ieta, const int iphi) const;

 private:
  // Typedef
  typedef unsigned int LUT;
  typedef std::vector<double> RCTdecompression;

  // Constant
  // TODO prefix k
  // Version 0 LUTs: 72 phis * 28 towers for HB/HE, 18*4 for HF
  static const int NUM_V0_LUTS = 2 * (72*28 + 18*4);
  // Version 1 LUTs: 72 phis * 2 for split 28,29 in HE, full granularity in HF
  static const int NUM_V1_LUTS = 2 * (72*2 + 36*12);
  static const int NOUTLUTS = NUM_V0_LUTS + NUM_V1_LUTS;
  static const unsigned int OUTPUT_LUT_SIZE = 1024;
  static const int TPGMAX = 256;

  // Member functions
  void loadHCALCompress(HcalLutMetadata const&, HcalTrigTowerGeometry const&) ; //Analytical compression tables
  void loadHCALCompress(const std::string& filename, HcalLutMetadata const&, HcalTrigTowerGeometry const&) ; //Compression tables from file
  void loadHCALUncompress(HcalLutMetadata const&, HcalTrigTowerGeometry const&) ; //Analytical decompression
  void loadHCALUncompress(const std::string& filename, HcalLutMetadata const&, HcalTrigTowerGeometry const&) ; //Decompression tables from file
  //int getLutGranularity(const DetId& id) const;
  //int getLutThreshold(const DetId& id) const;

  // Member Variables
  double nominal_gain_;
  double rctlsb_factor_;
  std::string compressionFile_;
  std::string decompressionFile_;
  std::vector<int> ietal;
  std::vector<int> ietah;
  std::vector<int> ZS;
  std::vector<int> LUTfactor;

  LUT *outputLUT_[NOUTLUTS];
  std::vector<RCTdecompression> hcaluncomp_;
};
#endif
