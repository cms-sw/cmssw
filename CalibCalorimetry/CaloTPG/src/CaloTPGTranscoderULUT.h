#ifndef CALOTPGTRANSCODERULUT_H
#define CALOTPGTRANSCODERULUT_H 1

#include <memory>
#include <vector>
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
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
  virtual const std::vector<unsigned int>& getCompressionLUT(const HcalTrigTowerDetId& id) const;
  virtual void setup(HcalLutMetadata const&, HcalTrigTowerGeometry const&, int, int);
  virtual int getOutputLUTId(const HcalTrigTowerDetId& id) const;
  virtual int getOutputLUTId(const int ieta, const int iphi) const;

 private:
  // Typedef
  typedef unsigned int LUT;
  typedef std::vector<double> RCTdecompression;

  const HcalTopology* theTopology;
  // Constant
  static const int NOUTLUTS = 4176;
  static const unsigned int OUTPUT_LUT_SIZE = 1024;
  static const unsigned int TPGMAX = 256;
  static const bool newHFphi = true;

  // Member functions
  void loadHCALCompress(HcalLutMetadata const&, HcalTrigTowerGeometry const&) ; //Analytical compression tables

  // Member Variables
  double nominal_gain_;
  double lsb_factor_;
  double rct_factor_;
  double nct_factor_;
  std::string compressionFile_;
  std::string decompressionFile_;
  std::vector<int> ietal;
  std::vector<int> ietah;
  std::vector<int> ZS;
  std::vector<int> LUTfactor;

  unsigned int size;
  std::vector<std::vector<LUT> > outputLUT_;
  std::vector<RCTdecompression> hcaluncomp_;
};
#endif
