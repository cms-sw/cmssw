#include "CalibFormats/CaloTPG/interface/HcalTPGTranscoder.h"

void HcalTPGTranscoder:: rctFineGrain(const HcalTriggerPrimitiveDigi& digi, std::vector<bool>& fineGrain) {
  fineGrain.resize(digi.size());
  for (int i=0; i<digi.size(); i++)
    fineGrain[i]=digi[i].fineGrain();
}
