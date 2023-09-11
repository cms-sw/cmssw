#include "DataFormats/EcalDigi/interface/EcalEBPhase2TriggerPrimitiveDigi.h"
#include <iostream>

EcalEBPhase2TriggerPrimitiveDigi::EcalEBPhase2TriggerPrimitiveDigi() : size_(0), data_(MAXSAMPLES) {}
//EcalTriggerPrimitiveDigi::EcalTriggerPrimitiveDigi(const EcalTrigTowerDetId& id) : id_(id),
//size_(0), data_(MAXSAMPLES) {
//}

EcalEBPhase2TriggerPrimitiveDigi::EcalEBPhase2TriggerPrimitiveDigi(const EBDetId& id) : id_(id), size_(0), data_(MAXSAMPLES) {}

void EcalEBPhase2TriggerPrimitiveDigi::setSample(int i, const EcalEBPhase2TriggerPrimitiveSample& sam) {
  //  std::cout << " In setSample  i " << i << "  sam " << sam << std::endl;
  data_[i] = sam;
  //  std::cout << " In setSample data_[i] " << data_[i] << std::endl;
}

int EcalEBPhase2TriggerPrimitiveDigi::sampleOfInterest() const {
  if (size_ == 1)
    return 0;
  else if (size_ == 5)
    return 2;
  else
    return -1;
}

/// get the encoded/compressed Et of interesting sample
int EcalEBPhase2TriggerPrimitiveDigi::encodedEt() const {
  int sample = sampleOfInterest();
  if (sample != -1)
    return data_[sample].encodedEt();
  else
    return -1;
}

bool EcalEBPhase2TriggerPrimitiveDigi::l1aSpike() const {
  int sample = sampleOfInterest();
  if (sample != -1)
    return data_[sample].l1aSpike();
  else
    return -1;
}

int EcalEBPhase2TriggerPrimitiveDigi::time() const {
  int sample = sampleOfInterest();
  if (sample != -1)
    return data_[sample].time();
  else
    return -1;
}

bool EcalEBPhase2TriggerPrimitiveDigi::isDebug() const {
  if (size_ == 1)
    return false;
  else if (size_ > 1)
    return true;
  return false;
}

void EcalEBPhase2TriggerPrimitiveDigi::setSize(int size) {
  if (size < 0)
    size_ = 0;
  else if (size > MAXSAMPLES)
    size_ = MAXSAMPLES;
  else
    size_ = size;
}

std::ostream& operator<<(std::ostream& s, const EcalEBPhase2TriggerPrimitiveDigi& digi) {
  s << digi.id() << " " << digi.size() << " samples " << std::endl;
  for (int i = 0; i < digi.size(); i++)
    s << "  " << digi.sample(i) << std::endl;
  return s;
}
