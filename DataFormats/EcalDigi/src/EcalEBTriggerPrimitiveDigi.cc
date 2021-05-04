#include "DataFormats/EcalDigi/interface/EcalEBTriggerPrimitiveDigi.h"
#include <iostream>

EcalEBTriggerPrimitiveDigi::EcalEBTriggerPrimitiveDigi() : size_(0), data_(MAXSAMPLES) {}
//EcalTriggerPrimitiveDigi::EcalTriggerPrimitiveDigi(const EcalTrigTowerDetId& id) : id_(id),
//size_(0), data_(MAXSAMPLES) {
//}

EcalEBTriggerPrimitiveDigi::EcalEBTriggerPrimitiveDigi(const EBDetId& id) : id_(id), size_(0), data_(MAXSAMPLES) {}

void EcalEBTriggerPrimitiveDigi::setSample(int i, const EcalEBTriggerPrimitiveSample& sam) {
  //  std::cout << " In setSample  i " << i << "  sam " << sam << std::endl;
  data_[i] = sam;
  //  std::cout << " In setSample data_[i] " << data_[i] << std::endl;
}

int EcalEBTriggerPrimitiveDigi::sampleOfInterest() const {
  if (size_ == 1)
    return 0;
  else if (size_ == 5)
    return 2;
  else
    return -1;
}

/// get the encoded/compressed Et of interesting sample
int EcalEBTriggerPrimitiveDigi::encodedEt() const {
  int sample = sampleOfInterest();
  if (sample != -1)
    return data_[sample].encodedEt();
  else
    return -1;
}

bool EcalEBTriggerPrimitiveDigi::l1aSpike() const {
  int sample = sampleOfInterest();
  if (sample != -1)
    return data_[sample].l1aSpike();
  else
    return -1;
}

int EcalEBTriggerPrimitiveDigi::time() const {
  int sample = sampleOfInterest();
  if (sample != -1)
    return data_[sample].time();
  else
    return -1;
}

bool EcalEBTriggerPrimitiveDigi::isDebug() const {
  if (size_ == 1)
    return false;
  else if (size_ > 1)
    return true;
  return false;
}

void EcalEBTriggerPrimitiveDigi::setSize(int size) {
  if (size < 0)
    size_ = 0;
  else if (size > MAXSAMPLES)
    size_ = MAXSAMPLES;
  else
    size_ = size;
}

std::ostream& operator<<(std::ostream& s, const EcalEBTriggerPrimitiveDigi& digi) {
  s << digi.id() << " " << digi.size() << " samples " << std::endl;
  for (int i = 0; i < digi.size(); i++)
    s << "  " << digi.sample(i) << std::endl;
  return s;
}
