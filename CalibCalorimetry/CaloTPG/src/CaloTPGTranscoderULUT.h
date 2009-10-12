#ifndef CALOTPGTRANSCODERULUT_H
#define CALOTPGTRANSCODERULUT_H 1

#include <vector>
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"

/** \class CaloTPGTranscoderULUT
  *  
  * $Date: 2008/09/05 05:06:24 $
  * $Revision: 1.12 $
  * \author J. Mans - Minnesota
  */
class CaloTPGTranscoderULUT : public CaloTPGTranscoder {
public:
  CaloTPGTranscoderULUT();
  CaloTPGTranscoderULUT(const std::string& hcalFile1, const std::string& hcalFile2);
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

 private:
  static const int NOUTLUTS = 4176;
  static const unsigned int OUTPUT_LUT_SIZE = 1024;
  static const int TPGMAX = 256;
  // Now introduce the zero-suppression
  static const int NR = 4;
  static const int ietal[NR];
  static const int ietah[NR];
  static const int ZS[NR];
  static const int LUTfactor[NR];
  static const double nominal_gain;
  static const double RCTLSB;
  static const bool newHFphi = true;
//
  void loadHCALCompress(void); //Analytical compression tables
  void loadHCALCompress(const std::string& filename); //Compression tables from file
  void loadHCALUncompress(void) const; //Analytical decompression
  void loadHCALUncompress(const std::string& filename) const; //Decompression tables from file
  virtual int GetOutputLUTId(const int ieta, const int iphi) const;

  typedef std::vector<unsigned char> LUTType;
  std::vector<LUTType> outputluts_;
  typedef unsigned char LUT;
  LUT *outputLUT[NOUTLUTS];
  unsigned int AnalyticalLUT[OUTPUT_LUT_SIZE];
  unsigned int IdentityLUT[OUTPUT_LUT_SIZE];
  typedef std::vector<double> RCTdecompression;
  mutable std::vector<RCTdecompression> hcaluncomp_;
  std::string DecompressionFile;
};

const int CaloTPGTranscoderULUT::ietal[NR] = { 1, 18, 27, 29};
const int CaloTPGTranscoderULUT::ietah[NR] = {17, 26, 28, 32};
const int CaloTPGTranscoderULUT::ZS[NR]    = { 4,  2,  1,  0};
const int CaloTPGTranscoderULUT::LUTfactor[NR] = { 1,  2,  5,  0};
const double CaloTPGTranscoderULUT::nominal_gain = 0.177;
const double CaloTPGTranscoderULUT::RCTLSB = 0.25;
#endif
