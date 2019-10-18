#ifndef CondFormatsHcalObjectsHcalTPParameters_h
#define CondFormatsHcalObjectsHcalTPParameters_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <algorithm>
#include <cstdint>

class HcalTPParameters {
public:
  HcalTPParameters();
  ~HcalTPParameters();

  // Load a new entry
  void loadObject(int version, int adcCut, uint64_t tdcMask, uint32_t tbits, int auxi1, int auxi2);

  /// get FineGrain Algorithm Version for HBHE
  int getFGVersionHBHE() const { return version_; }
  /// get ADC threshold fof TDC mask of HF
  int getADCThresholdHF() const { return adcCut_; }
  /// get TDC mask for HF
  uint64_t getTDCMaskHF() const { return tdcMask_; }
  /// get Self Trigger bits
  uint32_t getHFTriggerInfo() const { return tbits_; }
  /// get Axiliary words
  int getAuxi1() const { return auxi1_; }
  int getAuxi2() const { return auxi2_; }

private:
  int version_;
  int adcCut_;
  uint64_t tdcMask_;
  uint32_t tbits_;
  int auxi1_;
  int auxi2_;

  COND_SERIALIZABLE;
};

#endif
