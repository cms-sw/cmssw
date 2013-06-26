#include "DataFormats/HcalDigi/interface/HcalHistogramDigi.h"
#include <iomanip>

HcalHistogramDigi::HcalHistogramDigi() : id_(0) {
  for (int i=0; i<BINS_PER_HISTOGRAM*4; i++) 
    bins_[i]=0;
}

HcalHistogramDigi::HcalHistogramDigi(const HcalDetId& id) : id_(id) {
  for (int i=0; i<BINS_PER_HISTOGRAM*4; i++) 
    bins_[i]=0;
}

uint16_t HcalHistogramDigi::get(int capid, int bin) const {
  return bins_[(capid%4)*BINS_PER_HISTOGRAM+(bin%BINS_PER_HISTOGRAM)];
}

int HcalHistogramDigi::getSum(int bin) const {
  return (int)(bins_[(bin%BINS_PER_HISTOGRAM)])+
    (int)(bins_[BINS_PER_HISTOGRAM+(bin%BINS_PER_HISTOGRAM)])+
    (int)(bins_[BINS_PER_HISTOGRAM*2+(bin%BINS_PER_HISTOGRAM)])+
    (int)(bins_[BINS_PER_HISTOGRAM*3+(bin%BINS_PER_HISTOGRAM)]);
}

uint16_t* HcalHistogramDigi::getArray(int capid) {
  int offset=(capid%4)*BINS_PER_HISTOGRAM;
  return &(bins_[offset]);
}

std::ostream& operator<<(std::ostream& s, const HcalHistogramDigi& digi) {
  s << digi.id() << std::endl;
  for (int i=0; i<HcalHistogramDigi::BINS_PER_HISTOGRAM; i++) {
    s << ' ' << std::setw(2) << i;
    for (int capid=0; capid<4; capid++)
      s << std::setw(6) << digi.get(capid,i) << "  ";
    s << std::endl;
  }
  return s;
}
