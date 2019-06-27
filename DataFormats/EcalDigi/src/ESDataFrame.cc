#include "DataFormats/EcalDigi/interface/ESDataFrame.h"

ESDataFrame::ESDataFrame() : id_(0), size_(0) {}

ESDataFrame::ESDataFrame(const ESDetId& id) : id_(id), size_(0) {}

ESDataFrame::ESDataFrame(const edm::DataFrame& df) : id_(df.id()) {
  setSize(df.size());
  for (int i(0); i != size_; ++i) {
    static const int offset(65536);  // for uint16 to int16
    static const uint16_t limit(32767);
    const int dint(limit < df[i] ? (int)df[i] - offset : df[i]);
    data_[i] = ESSample((int16_t)dint);
  }
}

void ESDataFrame::setSize(int size) {
  if (size > MAXSAMPLES)
    size_ = MAXSAMPLES;
  else if (size <= 0)
    size_ = 0;
  else
    size_ = size;
}

std::ostream& operator<<(std::ostream& s, const ESDataFrame& digi) {
  s << digi.id() << " " << digi.size() << " samples " << std::endl;
  for (int i = 0; i < digi.size(); i++)
    s << "  " << digi.sample(i) << std::endl;
  return s;
}
