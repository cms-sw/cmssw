#ifndef HCALTBQADCUNPACKER_H
#define HCALTBQADCUNPACKER_H 1

#include "TBDataFormats/HcalTBObjects/interface/HcalTBBeamCounters.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

namespace hcaltb {
class HcalTBQADCUnpacker {
public:
  HcalTBQADCUnpacker();
  void unpack(const FEDRawData& raw,
	      HcalTBBeamCounters& beamadc, bool is04_=true) const;
private:
  bool isTB04_;
};

}

#endif
