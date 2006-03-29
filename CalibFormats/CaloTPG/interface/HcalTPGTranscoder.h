#ifndef CALIBFORMATS_CALOTPG_HCALTPGTRANSCODER_H
#define CALIBFORMATS_CALOTPG_HCALTPGTRANSCODER_H 1

#include "CalibFormats/CaloObjects/interface/IntegerCaloSamples.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"
#include <vector>

/** \class HcalTPGTranscoder
  *  
  * Abstract interface for the mutual transcoder required for compressing
  * and uncompressing the ET stored in HCAL Trigger Primitives
  * 
  * $Date: $
  * $Revision: $
  * \author J. Mans - Minnesota
  */
class HcalTPGTranscoder {
public:
  /** \brief Compression from linear samples+fine grain in the HTR */
  virtual void htrCompress(const IntegerCaloSamples& ics, const std::vector<bool>& fineGrain, HcalTriggerPrimitiveDigi& digi) = 0;
  /** \brief Uncompression for the Electron/Photon path in the RCT */
  virtual void rctEGammaUncompress(const HcalTriggerPrimitiveDigi& digi, IntegerCaloSamples& ics) = 0;
  /** \brief Uncompression for the JET path in the RCT */
  virtual void rctJetUncompress(const HcalTriggerPrimitiveDigi& digi, IntegerCaloSamples& ics) = 0;
  /** \brief Extract the fine-grain bits from this digi.
      \note This does not need to be virtual since it is a defined characteristic of the digis
  */
  void rctFineGrain(const HcalTriggerPrimitiveDigi& digi, std::vector<bool>& fineGrain);
};

#endif
