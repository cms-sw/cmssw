#include "CalibFormats/CaloTPG/interface/HcalTPGTranscoder.h"

const int HcalTPGTranscoder::EGAMMA_LUT_SIZE=128;
const int HcalTPGTranscoder::JET_LUT_SIZE=256;

void HcalTPGTranscoder:: rctFineGrain(const HcalTriggerPrimitiveDigi& digi, std::vector<bool>& fineGrain) {
  fineGrain.resize(digi.size());
  for (int i=0; i<digi.size(); i++)
    fineGrain[i]=digi[i].fineGrain();
}
