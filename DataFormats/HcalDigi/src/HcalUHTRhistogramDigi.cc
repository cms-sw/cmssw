#include "DataFormats/HcalDigi/interface/HcalUHTRhistogramDigi.h"
#include <iomanip>

HcalUHTRhistogramDigi::HcalUHTRhistogramDigi(int nbins, bool sepCapIds) : id_(0) {
  std::vector<uint32_t> reserve;
    for (int j=0; j<nbins; j++) 
      reserve.push_back(0);
    for (int i=0; i<=3*sepCapIds; i++)
      histo_.push_back(reserve);
    
    nb_=nbins;
    sc_=sepCapIds;
}
HcalUHTRhistogramDigi::HcalUHTRhistogramDigi(int nbins, bool sepCapIds, const HcalDetId& id) : id_(id) {
  std::vector<uint32_t> reserve;
    for (int j=0; j<nbins; j++) 
      reserve.push_back(0);
    for (int i=0; i<=3*sepCapIds; i++)
      histo_.push_back(reserve);
    nb_=nbins;
    sc_=sepCapIds;
}

uint32_t HcalUHTRhistogramDigi::get(int capid, int bin) const {
  return histo_[capid][bin];  //not segfault proof
}

int HcalUHTRhistogramDigi::getSum(int bin) const {
  int capSum = 0;
  for (unsigned int cap=0; cap<histo_.size(); cap++)
    capSum+=histo_[cap][bin];
  return capSum;
}

bool HcalUHTRhistogramDigi::fillBin(int capid, int bin, uint32_t val) {
  if (histo_[capid][bin] == 0) {
    histo_[capid][bin] = val;
    return true;
  }
  else return false;
}

std::ostream& operator<<(std::ostream& s, const HcalUHTRhistogramDigi& digi) {
  s << digi.id() << std::endl;
  for (int i=0; i<=3*digi.sc(); i++) {
    s << ' ' << std::setw(2) << i;
    for (int j=0; j<digi.nb(); j++)
      s << std::setw(6) << digi.get(i, j) << "  ";
    s << std::endl;
  }
  return s;
}
