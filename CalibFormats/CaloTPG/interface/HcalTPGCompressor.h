#ifndef CalibFormats_CaloTPG_HcalTPGCompressor_h
#define CalibFormats_CaloTPG_HcalTPGCompressor_h

/** \class HcalTPGCompressor
  *  
  * \author J. Mans - Minnesota
  */

#include "CalibFormats/CaloObjects/interface/IntegerCaloSamples.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveSample.h"

#include <vector>

class CaloTPGTranscoder;

class HcalTPGCompressor {
public:
  HcalTPGCompressor(const CaloTPGTranscoder* coder);
  void compress(const IntegerCaloSamples& ics, const std::vector<int>& fineGrain, HcalTriggerPrimitiveDigi& digi) const;
  HcalTriggerPrimitiveSample compress(const HcalTrigTowerDetId& id, unsigned int sample, bool fineGrain) const;

private:
  const CaloTPGTranscoder* coder_;
};

#endif
