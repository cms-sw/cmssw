#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/EcalTPGCompressor.h"

EcalTPGCompressor::EcalTPGCompressor(const CaloTPGTranscoder* coder) : coder_(coder) {}

void EcalTPGCompressor::compress(const IntegerCaloSamples& ics,
                                 const std::vector<bool>& fineGrain,
                                 EcalTriggerPrimitiveDigi& digi) const {
  digi.setSize(ics.size());
  for (int i = 0; i < ics.size(); i++)
    digi.setSample(i, coder_->ecalCompress(ics.id(), ics[i], fineGrain[i]));
}

EcalTriggerPrimitiveSample EcalTPGCompressor::compress(const EcalTrigTowerDetId& id,
                                                       unsigned int sample,
                                                       bool fineGrain) const {
  return coder_->ecalCompress(id, sample, fineGrain);
}
