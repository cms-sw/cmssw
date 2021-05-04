#include "DataFormats/HcalRecHit/interface/HBHEChannelInfo.h"

namespace {
  template <typename T>
  void dumpArray(std::ostream& s, const char* p, const T* arr, const unsigned len) {
    s << ' ' << p;
    for (unsigned i = 0; i < len; ++i) {
      s << ' ' << *arr++;
    }
  }

  template <typename T>
  void dumpArrayAsUnsigned(std::ostream& s, const char* p, const T* arr, const unsigned len) {
    s << ' ' << p;
    for (unsigned i = 0; i < len; ++i) {
      s << ' ' << static_cast<unsigned>(*arr++);
    }
  }
}  // namespace

std::ostream& operator<<(std::ostream& s, const HBHEChannelInfo& inf) {
  const unsigned nSamples = inf.nSamples();

  s << inf.id() << " :"
    << " recoShape " << inf.recoShape() << " nSamples " << nSamples << " soi " << inf.soi() << " capid " << inf.capid()
    << " hasTDC " << inf.hasTimeInfo() << " hasEffPeds " << inf.hasEffectivePedestals() << " dropped "
    << inf.isDropped() << " linkErr " << inf.hasLinkError() << " capidErr " << inf.hasCapidError() << " darkI "
    << inf.darkCurrent() << " fcByPE " << inf.fcByPE() << " lambda " << inf.lambda();
  dumpArray(s, "rawCharge", inf.rawCharge(), nSamples);
  dumpArray(s, "peds", inf.pedestal(), nSamples);
  dumpArray(s, "noise", inf.pedestalWidth(), nSamples);
  dumpArray(s, "gain", inf.gain(), nSamples);
  dumpArray(s, "gainWidth", inf.gainWidth(), nSamples);
  dumpArray(s, "dFcPerADC", inf.dFcPerADC(), nSamples);
  dumpArrayAsUnsigned(s, "adc", inf.adc(), nSamples);
  if (inf.hasTimeInfo()) {
    dumpArray(s, "tdc", inf.riseTime(), nSamples);
  }

  return s;
}
