#include "DataFormats/HcalDigi/interface/CastorTriggerPrimitiveDigi.h"

CastorTriggerPrimitiveDigi::CastorTriggerPrimitiveDigi() : size_(0), hcalPresamples_(0) {}
CastorTriggerPrimitiveDigi::CastorTriggerPrimitiveDigi(const HcalCastorDetId& id)
    : id_(id), size_(0), hcalPresamples_(0) {}

void CastorTriggerPrimitiveDigi::setSize(int size) {
  if (size < 0)
    size_ = 0;
  else if (size > MAXSAMPLES)
    size_ = MAXSAMPLES;
  else
    size_ = size;
}
void CastorTriggerPrimitiveDigi::setPresamples(int ps) {
  if (ps < 0)
    hcalPresamples_ &= 0xFFFFFF0;
  //  else if (ps>=size_) hcalPresamples_=size_-1;
  else
    hcalPresamples_ |= ps & 0xF;
}

void CastorTriggerPrimitiveDigi::setZSInfo(bool unsuppressed, bool markAndPass) {
  if (markAndPass)
    hcalPresamples_ |= 0x10;
  if (unsuppressed)
    hcalPresamples_ |= 0x20;
}

std::ostream& operator<<(std::ostream& s, const CastorTriggerPrimitiveDigi& digi) {
  s << digi.id() << " " << digi.size() << " samples " << digi.presamples() << " presamples";
  if (digi.zsUnsuppressed())
    s << " zsUS";
  if (digi.zsMarkAndPass())
    s << " zsM&P";
  s << std::endl;
  s << " SOI  tpchannel=" << digi.SOI_tpchannel() << " tpdata 0x" << std::hex << digi.SOI_tpdata() << std::dec
    << std::endl;
  for (int i = 0; i < digi.size(); i++) {
    s << "  0x" << std::hex << digi.sample(i).raw() << " tpdata 0x" << digi.tpdata(i) << std::dec
      << " channel=" << digi.tpchannel(i);
    if (digi.isSOI(i))
      s << " SOI";
    s << std::endl;
  }
  return s;
}
