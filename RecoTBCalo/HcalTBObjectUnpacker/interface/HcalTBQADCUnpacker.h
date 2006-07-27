#ifndef HCALTBQADCUNPACKER_H
#define HCALTBQADCUNPACKER_H 1
using namespace std;
#include "TBDataFormats/HcalTBObjects/interface/HcalTBBeamCounters.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

namespace hcaltb {
class HcalTBQADCUnpacker {
public:
  HcalTBQADCUnpacker();
  void unpack(const FEDRawData& raw,
	      HcalTBBeamCounters& beamadc, bool is04_=true) const;
  void setCalib(const vector<vector<string> >& calibLines_);
private:
  bool isTB04_;
  double qdc_ped[128];
  double qdc_gain[128];
};

}

#endif
