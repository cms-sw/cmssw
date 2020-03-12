#include "DataFormats/EcalDigi/interface/EcalPnDiodeDigi.h"

EcalPnDiodeDigi::EcalPnDiodeDigi() : size_(0), data_(MAXSAMPLES) {}
EcalPnDiodeDigi::EcalPnDiodeDigi(const EcalPnDiodeDetId& id) : id_(id), size_(0), data_(MAXSAMPLES) {}

void EcalPnDiodeDigi::setSize(int size) {
  if (size < 0)
    size_ = 0;
  else if (size > MAXSAMPLES)
    size_ = MAXSAMPLES;
  else
    size_ = size;
}

std::ostream& operator<<(std::ostream& s, const EcalPnDiodeDigi& digi) {
  s << digi.id() << " " << digi.size() << " samples " << std::endl;
  for (int i = 0; i < digi.size(); i++)
    s << "  " << digi.sample(i) << std::endl;
  return s;
}
