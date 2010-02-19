#ifndef CALOTPGTRANSCODERULUT_H
#define CALOTPGTRANSCODERULUT_H 1

#include <vector>
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"

/** \class CaloTPGTranscoderULUT
  *  
  * $Date: 2009/05/29 07:25:08 $
  * $Revision: 1.13 $
  * \author J. Mans - Minnesota
  */
class CaloTPGTranscoderULUT : public CaloTPGTranscoder {
public:
  CaloTPGTranscoderULUT();
  CaloTPGTranscoderULUT(const std::string& hcalFile1, const std::string& hcalFile2);
  CaloTPGTranscoderULUT(const std::vector<int>& _ietal,const std::vector<int>& _ietah,const std::vector<int>& _zs,const std::vector<int>& _lutfactor, const double& _rctlsb, const double& _nominalgain, const std::string& hcalFile1, const std::string& hcalFile2);
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
  std::vector<int> ietal;
  std::vector<int> ietah;
  std::vector<int> ZS;
  std::vector<int> LUTfactor;
  double nominal_gain;
  double RCTLSB;
  double RCTLSB_factor;
  int NR;
  static const bool newHFphi = true;

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

  void setLUTGranularity( const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, const std::vector<int>& );
  void setRCTLSB(const double&);
  void setNominalGain(const double&);

};
#endif
