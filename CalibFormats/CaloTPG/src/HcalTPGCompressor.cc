#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/HcalTPGCompressor.h"

HcalTPGCompressor::HcalTPGCompressor(const CaloTPGTranscoder* coder) : coder_(coder) {}

void HcalTPGCompressor::compress(const IntegerCaloSamples& ics,
                                 const std::vector<int>& fineGrain,
                                 HcalTriggerPrimitiveDigi& digi) const {
  digi.setSize(ics.size());
  digi.setPresamples(ics.presamples());
  for (int i = 0; i < ics.size(); i++)
    digi.setSample(i, coder_->hcalCompress(ics.id(), ics[i], fineGrain[i]));
}

HcalTriggerPrimitiveSample HcalTPGCompressor::compress(const HcalTrigTowerDetId& id,
                                                       unsigned int sample,
                                                       bool fineGrain) const {
  return coder_->hcalCompress(id, sample, fineGrain);
}
