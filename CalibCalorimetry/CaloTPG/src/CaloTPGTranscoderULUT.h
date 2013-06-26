#ifndef CALOTPGTRANSCODERULUT_H
#define CALOTPGTRANSCODERULUT_H 1

#include <vector>
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"

// tmp
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/HcalObjects/interface/HcalLutMetadata.h"


/** \class CaloTPGTranscoderULUT
  *  
  * $Date: 2012/08/28 13:40:07 $
  * $Revision: 1.17 $
  * \author J. Mans - Minnesota
  */

class HcalTrigTowerGeometry;

class CaloTPGTranscoderULUT : public CaloTPGTranscoder {
public:
  CaloTPGTranscoderULUT(const std::string& compressionFile="",
                        const std::string& decompressionFile="");
  virtual ~CaloTPGTranscoderULUT();
  virtual HcalTriggerPrimitiveSample hcalCompress(const HcalTrigTowerDetId& id, unsigned int sample, bool fineGrain) const;
  virtual EcalTriggerPrimitiveSample ecalCompress(const EcalTrigTowerDetId& id, unsigned int sample, bool fineGrain) const;

  virtual void rctEGammaUncompress(const HcalTrigTowerDetId& hid, const HcalTriggerPrimitiveSample& hc,
				   const EcalTrigTowerDetId& eid, const EcalTriggerPrimitiveSample& ec, 
				   unsigned int& et, bool& egVecto, bool& activity) const;
  virtual void rctJetUncompress(const HcalTrigTowerDetId& hid, const HcalTriggerPrimitiveSample& hc,
				   const EcalTrigTowerDetId& eid, const EcalTriggerPrimitiveSample& ec, 
				   unsigned int& et) const;
  virtual double hcaletValue(const int& ieta, const int& compressedValue) const;
  virtual double hcaletValue(const int& ieta, const int& iphi, const int& compressedValue) const;
  virtual double hcaletValue(const HcalTrigTowerDetId& hid, const HcalTriggerPrimitiveSample& hc) const;
  virtual bool HTvalid(const int ieta, const int iphi) const;
  virtual std::vector<unsigned char> getCompressionLUT(HcalTrigTowerDetId id) const;
  virtual void setup(const edm::EventSetup& es, Mode) const;
  virtual int getOutputLUTId(const int ieta, const int iphi) const;
  void printDecompression() const;

 private:
  // Typedef
  typedef unsigned int LUT;
  typedef std::vector<double> RCTdecompression;

  // Constant
  // TODO prefix k
  static const int NOUTLUTS = 4176;
  static const unsigned int OUTPUT_LUT_SIZE = 1024;
  static const int TPGMAX = 256;
  static const bool newHFphi = true;

  // Member functions
  void loadHCALCompress(void) const; //Analytical compression tables
  void loadHCALCompress(const std::string& filename) const; //Compression tables from file
  void loadHCALUncompress(void) const; //Analytical decompression
  void loadHCALUncompress(const std::string& filename) const; //Decompression tables from file
  //int getLutGranularity(const DetId& id) const;
  //int getLutThreshold(const DetId& id) const;

  // Member Variables
  mutable bool isLoaded_;
  mutable double nominal_gain_;
  mutable double rctlsb_factor_;
  std::string compressionFile_;
  std::string decompressionFile_;
  std::vector<int> ietal;
  std::vector<int> ietah;
  std::vector<int> ZS;
  std::vector<int> LUTfactor;

  mutable LUT *outputLUT_[NOUTLUTS];
  mutable std::vector<RCTdecompression> hcaluncomp_;
  mutable edm::ESHandle<HcalLutMetadata> lutMetadata_;
  mutable edm::ESHandle<HcalTrigTowerGeometry> theTrigTowerGeometry;
};
#endif
