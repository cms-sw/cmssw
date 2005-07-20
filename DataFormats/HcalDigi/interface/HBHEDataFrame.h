#ifndef DIGIHCAL_HBHEDATAFRAME_H
#define DIGIHCAL_HBHEDATAFRAME_H

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"
#include <iostream>

namespace cms {

class HBHEDataFrame {
public:
  HBHEDataFrame(); // for persistence
  HBHEDataFrame(const HcalDetId& id);

  const HcalDetId& id() const { return id_; }
  const HcalElectronicsId& elecId() const { return electronicsId_; }

  int size() const { return size_; }
  int presamples() const { return hcalPresamples_; }

  const HcalQIESample& operator[](int i) const { return data_[i]; }
  const HcalQIESample& sample(int i) const { return data_[i]; }

  void setSize(int size);
  void setPresamples(int ps);
  void setSample(int i, const HcalQIESample& sam) { data_[i]=sam; }
  void setReadoutIds(const HcalElectronicsId& eid);

  static const int MAXSAMPLES = 10;
private:
  HcalDetId id_;
  HcalElectronicsId electronicsId_; 
  int size_;
  int hcalPresamples_;
  HcalQIESample data_[10];
};

}

std::ostream& operator<<(std::ostream&, const cms::HBHEDataFrame&);

#endif
