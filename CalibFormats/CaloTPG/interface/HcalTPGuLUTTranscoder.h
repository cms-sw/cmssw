#ifndef CALIBFORMATS_CALOTPG_HCALTPGULUTTRANSCODER_H
#define CALIBFORMATS_CALOTPG_HCALTPGULUTTRANSCODER_H 1
#include <iostream.h>
#include <fstream.h>
#include "CalibFormats/CaloTPG/interface/HcalTPGTranscoder.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"

/** \class HcalTPGSimpleTranscoder
  *  
  * Simple transcoder which matches energy scales.
  * 
  * Codes 0->127 use the EGamma scale
  * Codes 128->128+128*(escale/jscale) use the Jet scale
  * Codes 128+... ->255 use 2*Jet scale
  *
  * $Date: 2006/04/03 20:57:07 $
  * $Revision: 1.1 $
  * \author J. Mans - Minnesota
  */
class HcalTPGuLUTTranscoder : public HcalTPGTranscoder {
public:
  HcalTPGuLUTTranscoder(int hcalMeV, int egammaMeV, int jetMeV);
  virtual ~HcalTPGuLUTTranscoder();
  virtual void htrCompress(const IntegerCaloSamples& ics, const std::vector<bool>& fineGrain, HcalTriggerPrimitiveDigi& digi);
  virtual void rctEGammaUncompress(const HcalTriggerPrimitiveDigi& digi, IntegerCaloSamples& ics);
  virtual void rctJetUncompress(const HcalTriggerPrimitiveDigi& digi, IntegerCaloSamples& ics);
  void printTable();
  void filluLUT();
private:
  void buildTable();
  //  void filluLUT();
  int hcalLSBMeV_;
  int egammaLSBMeV_;
  int jetLSBMeV_;
  int LUT_[256][5];
  std::vector<uint8_t> hcalTable_;
  std::vector<uint8_t> egammaTable_;
  std::vector<uint8_t> jetTable_;
  std::vector<int> codeMeV_; // for testing only
};

#endif
