#ifndef CALOTPGTRANSCODERULUT_H
#define CALOTPGTRANSCODERULUT_H 1

#include <vector>
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"

/** \class CaloTPGTranscoderULUT
  *  
  * $Date: 2007/01/04 19:33:17 $
  * $Revision: 1.9 $
  * \author J. Mans - Minnesota
  */
class CaloTPGTranscoderULUT : public CaloTPGTranscoder {
public:
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

  void loadhcalUncompress();
  virtual double hcaletValue(const int& ieta, const int& compressedValue) const;
  virtual double hcaletValue(const HcalTrigTowerDetId& hid, const HcalTriggerPrimitiveSample& hc) const;

 private:
  static const int N_TOWER = 32, noutluts = 4176;
  static const unsigned int OUTPUT_LUT_SIZE = 1024;
  typedef std::vector<unsigned char> LUTType;
  std::vector<LUTType> hcal_;
  std::vector<const LUTType*> hcalITower_;
  void loadHCAL(const std::string& filename);
  void loadhcalUncompress(const std::string& filename);
  virtual bool HTvalid(const int ieta, const int iphi) const;
  virtual int GetOutputLUTId(const int ieta, const int iphi) const;
  std::vector<LUTType> outputluts_;
  typedef unsigned char LUT;
  LUT *outputLUT[noutluts];  
  double hcaluncomp_[33][256] ;
};
#endif
