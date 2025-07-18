#ifndef DataFormats_HcalDigi_HcalDigiHostCollection_h
#define DataFormats_HcalDigi_HcalDigiHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/HcalDigi/interface/HcalDigiSoA.h"

namespace hcal {

  // HcalDigiSoA in host memory
  using Phase1DigiHostCollection = PortableHostCollection<HcalPhase1DigiSoA>;
  using Phase0DigiHostCollection = PortableHostCollection<HcalPhase0DigiSoA>;
}  // namespace hcal

#endif
