#ifndef CALOTPGTRANSCODERULUT_H
#define CALOTPGTRANSCODERULUT_H 1

#include <vector>
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"

/** \class CaloTPGTranscoderULUT
  *  
  * $Date: 2006/12/13 18:10:35 $
  * $Revision: 1.6 $
  * \author J. Mans - Minnesota
  */
class CaloTPGTranscoderULUT : public CaloTPGTranscoder {
public:
  CaloTPGTranscoderULUT(const std::string& hcalFile1, const std::string& hcalFile2);
  virtual HcalTriggerPrimitiveSample hcalCompress(const HcalTrigTowerDetId& id, unsigned int sample, bool fineGrain) const;
  virtual EcalTriggerPrimitiveSample ecalCompress(const EcalTrigTowerDetId& id, unsigned int sample, bool fineGrain) const;

  virtual void rctEGammaUncompress(const HcalTrigTowerDetId& hid, const HcalTriggerPrimitiveSample& hc,
				   const EcalTrigTowerDetId& eid, const EcalTriggerPrimitiveSample& ec, 
				   unsigned int& et, bool& egVecto, bool& activity) const;
  virtual void rctJetUncompress(const HcalTrigTowerDetId& hid, const HcalTriggerPrimitiveSample& hc,
				   const EcalTrigTowerDetId& eid, const EcalTriggerPrimitiveSample& ec, 
				   unsigned int& et) const;
  virtual double hcaletValue(const int& ieta, const int& compET) const;
  
 private:
  static const int N_TOWER = 32;
  typedef std::vector<int> LUTType;
  std::vector<LUTType> hcal_;
  std::vector<const LUTType*> hcalITower_;
  void loadHCAL(const std::string& filename);
  void loadhcalUncompress(const std::string& filename);
  
  double hcaluncomp_[32][256];
};
#endif
