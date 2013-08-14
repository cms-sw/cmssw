#ifndef IORAWDATA_CALOPATTERNS_HCALFIBERPATTERN_H
#define IORAWDATA_CALOPATTERNS_HCALFIBERPATTERN_H 1

#include "DataFormats/HcalDigi/interface/HcalQIESample.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include <string>
#include <map>
#include <vector>

/** \class HcalFiberPattern
  *  
  * $Date: 2006/09/29 17:57:39 $
  * $Revision: 1.1 $
  * \author J. Mans - Minnesota
  */
class HcalFiberPattern {
public:
  HcalFiberPattern(const std::map<std::string, std::string>& params, const std::vector<uint32_t>& data);
  std::vector<HcalQIESample> getSamples(int bunch, int npresamples, int nsamples, int fiberChan);
  HcalElectronicsId getId(int fiberChan);
  int crate() const { return crate_; }
  int slot() const { return slot_; }
  int fiber() const { return fiber_; }
private:
  HcalQIESample unpack(int bc, int fc);
  int crate_, slot_, tb_, fiber_, spigot_, dcc_;
  std::vector<uint32_t> pattern_;
};

#endif
