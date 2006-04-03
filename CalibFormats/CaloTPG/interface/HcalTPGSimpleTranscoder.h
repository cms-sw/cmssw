#ifndef CALIBFORMATS_CALOTPG_HCALTPGSIMPLETRANSCODER_H
#define CALIBFORMATS_CALOTPG_HCALTPGSIMPLETRANSCODER_H 1

#include "CalibFormats/CaloTPG/interface/HcalTPGTranscoder.h"

/** \class HcalTPGSimpleTranscoder
  *  
  * Simple transcoder which matches energy scales.
  * 
  * Codes 0->127 use the EGamma scale
  * Codes 128->128+128*(escale/jscale) use the Jet scale
  * Codes 128+... ->255 use 2*Jet scale
  *
  * $Date: $
  * $Revision: $
  * \author J. Mans - Minnesota
  */
class HcalTPGSimpleTranscoder : public HcalTPGTranscoder {
public:
  HcalTPGSimpleTranscoder(int hcalMeV, int egammaMeV, int jetMeV);
  virtual ~HcalTPGSimpleTranscoder();
  virtual void htrCompress(const IntegerCaloSamples& ics, const std::vector<bool>& fineGrain, HcalTriggerPrimitiveDigi& digi);
  virtual void rctEGammaUncompress(const HcalTriggerPrimitiveDigi& digi, IntegerCaloSamples& ics);
  virtual void rctJetUncompress(const HcalTriggerPrimitiveDigi& digi, IntegerCaloSamples& ics);
  void printTable();
private:
  void buildTable();
  int hcalLSBMeV_;
  int egammaLSBMeV_;
  int jetLSBMeV_;
  std::vector<uint8_t> hcalTable_;
  std::vector<uint8_t> egammaTable_;
  std::vector<uint8_t> jetTable_;
  std::vector<int> codeMeV_; // for testing only
};

#endif
