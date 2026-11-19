#ifndef CalibFormats_CaloTPG_CaloTPGTranscoder_h
#define CalibFormats_CaloTPG_CaloTPGTranscoder_h

/** \class CaloTPGTranscoder
  *  
  * Abstract interface for the mutual transcoder required for compressing
  * and uncompressing the ET stored in HCAL and ECAL Trigger Primitives
  * 
  * \author J. Mans - Minnesota
  */

#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveSample.h"

#include <memory>

class HcalTPGCompressor;
class EcalTPGCompressor;

class CaloTPGTranscoder {
public:
  CaloTPGTranscoder();
  virtual ~CaloTPGTranscoder();

  /** \brief Compression from linear samples+fine grain in the HTR */
  virtual HcalTriggerPrimitiveSample hcalCompress(const HcalTrigTowerDetId& id,
                                                  unsigned int sample,
                                                  int fineGrain) const = 0;
  /** \brief Compression from linear samples+fine grain in the ECAL */
  virtual EcalTriggerPrimitiveSample ecalCompress(const EcalTrigTowerDetId& id,
                                                  unsigned int sample,
                                                  bool fineGrain) const = 0;

  virtual double hcaletValue(const int& ieta, const int& iphi, const int& version, const int& compressedValue) const = 0;
  virtual double hcaletValue(const HcalTrigTowerDetId& hid, const HcalTriggerPrimitiveSample& hc) const = 0;
  std::shared_ptr<const HcalTPGCompressor> getHcalCompressor() const { return hccompress_; }
  std::shared_ptr<const EcalTPGCompressor> getEcalCompressor() const { return eccompress_; }

private:
  std::shared_ptr<const HcalTPGCompressor> hccompress_;
  std::shared_ptr<const EcalTPGCompressor> eccompress_;
};

#endif
