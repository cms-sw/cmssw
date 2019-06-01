#include "IORawData/CaloPatterns/interface/HcalFiberPattern.h"
#include "FWCore/Utilities/interface/Exception.h"

static inline int setIf(const std::string& name, const std::map<std::string, std::string>& params) {
  std::map<std::string, std::string>::const_iterator j = params.find(name);
  if (j == params.end())
    throw cms::Exception("InvalidFormat") << "Missing parameter '" << name << "'";
  else
    return strtol(j->second.c_str(), nullptr, 0);
}

HcalFiberPattern::HcalFiberPattern(const std::map<std::string, std::string>& params, const std::vector<uint32_t>& data)
    : pattern_(data) {
  crate_ = setIf("CRATE", params);
  slot_ = setIf("SLOT", params);
  fiber_ = setIf("FIBER", params);
  dcc_ = setIf("DCC", params);
  spigot_ = setIf("SPIGOT", params);
  tb_ = setIf("TOPBOTTOM", params);
}

HcalQIESample HcalFiberPattern::unpack(int bc, int fc) {
  uint32_t w1 = pattern_[bc * 2];      // lsw
  uint32_t w2 = pattern_[bc * 2 + 1];  // msw

  int adc = 0, capid = 0;
  bool dv = (w1 & 0x10000) != 0;
  bool er = (w1 & 0x20000) != 0;

  switch (fc) {
    case (0):
      adc = (w2 & 0xFE00) >> 9;
      capid = (w1 & 0x0180) >> 7;
      break;
    case (1):
      adc = (w2 & 0xFE) >> 1;
      capid = (w1 & 0x0060) >> 5;
      break;
    case (2):
      adc = (w1 & 0xFE00) >> 9;
      capid = (w1 & 0x0018) >> 3;
      break;
    default:
      break;
  }
  return HcalQIESample(adc, capid, fiber_, fc, dv, er);
}

std::vector<HcalQIESample> HcalFiberPattern::getSamples(int bunch, int npresamples, int nsamples, int fiberChan) {
  if (bunch < npresamples)
    throw cms::Exception("InvalidArgument")
        << "Asked for " << npresamples << " presamples with event at bunch " << bunch;
  if (nsamples - npresamples + bunch >= (int)(pattern_.size() / 2))
    throw cms::Exception("InvalidArgument")
        << "Asked for " << nsamples << " with event at " << bunch << " and " << npresamples << " presamples, but only "
        << pattern_.size() / 2 << " bunches are available";

  std::vector<HcalQIESample> retval;
  retval.reserve(nsamples);

  for (int i = 0; i < nsamples; i++) {
    int bc = bunch + i - npresamples;
    retval.push_back(unpack(bc, fiberChan));
  }
  return retval;
}

HcalElectronicsId HcalFiberPattern::getId(int fiberChan) {
  HcalElectronicsId retval(fiberChan, fiber_, spigot_, dcc_);
  retval.setHTR(crate_, slot_, tb_);
  return retval;
}
