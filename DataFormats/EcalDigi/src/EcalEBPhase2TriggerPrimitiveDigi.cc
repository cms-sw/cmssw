#include "DataFormats/EcalDigi/interface/EcalEBPhase2TriggerPrimitiveDigi.h"
#include <iostream>
#include <algorithm>

static constexpr int MAXSAMPLES = 20;
// This is the number of digi samples one wants to put in the TP digis. In Phase I was 5, but this setting was used. In the end in Phase I one one sample, corresponding to BC0 was filled. For Phase2 we have not decided yet. We leave this setting as it is while we decide what to do

EcalEBPhase2TriggerPrimitiveDigi::EcalEBPhase2TriggerPrimitiveDigi() : size_(0), data_(MAXSAMPLES) {}

EcalEBPhase2TriggerPrimitiveDigi::EcalEBPhase2TriggerPrimitiveDigi(const EBDetId& id)
    : id_(id), size_(0), data_(MAXSAMPLES) {}

void EcalEBPhase2TriggerPrimitiveDigi::setSample(int i, const EcalEBPhase2TriggerPrimitiveSample& sam) {
  data_[i] = sam;
}

int EcalEBPhase2TriggerPrimitiveDigi::sampleOfInterest() const {
  // sample  of interest to be save in the TP digis
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

bool EcalEBPhase2TriggerPrimitiveDigi::isDebug() const { return (size_ > 1); }

void EcalEBPhase2TriggerPrimitiveDigi::setSize(int size) { size_ = std::clamp(size_, 0, MAXSAMPLES); }

std::ostream& operator<<(std::ostream& s, const EcalEBPhase2TriggerPrimitiveDigi& digi) {
  s << digi.id() << " " << digi.size() << " samples " << std::endl;
  for (int i = 0; i < digi.size(); i++)
    s << "  " << digi.sample(i) << std::endl;
  return s;
}
