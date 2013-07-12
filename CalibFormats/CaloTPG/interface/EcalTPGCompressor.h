#ifndef CALIBFORMATS_CALOTPG_ECALTPGCOMPRESSOR_H
#define CALIBFORMATS_CALOTPG_ECALTPGCOMPRESSOR_H 1

#include "CalibFormats/CaloObjects/interface/IntegerCaloSamples.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h"
class CaloTPGTranscoder;

/** \class EcalTPGCompressor
  *  
  * $Date: $
  * $Revision: $
  * \author J. Mans - Minnesota
  */
class EcalTPGCompressor {
public:
  EcalTPGCompressor(const CaloTPGTranscoder* coder);
  void compress(const IntegerCaloSamples& ics, const std::vector<bool>& fineGrain, EcalTriggerPrimitiveDigi& digi) const;
  EcalTriggerPrimitiveSample compress(const EcalTrigTowerDetId& id, unsigned int sample, bool fineGrain) const;
private:
  const CaloTPGTranscoder* coder_;
};

#endif
