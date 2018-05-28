#include "DataFormats/EcalDigi/interface/EcalEBClusterTriggerPrimitiveDigi.h"
#include <iostream>


EcalEBClusterTriggerPrimitiveDigi::EcalEBClusterTriggerPrimitiveDigi() : size_(0), data_(MAXSAMPLES) {
}

EcalEBClusterTriggerPrimitiveDigi::EcalEBClusterTriggerPrimitiveDigi(const EBDetId& tpid, const std::vector<EBDetId>& xtalIds, float etaClu, float phiClu) : 
  tpId_(tpid), 
  cryIdInCluster_(xtalIds),  
  etaClu_(etaClu),
  phiClu_(phiClu),
  size_(0), data_(MAXSAMPLES) {
}

void EcalEBClusterTriggerPrimitiveDigi::setSample(int i, const EcalEBClusterTriggerPrimitiveSample& sam) 
{

  data_[i]=sam;

  
}

int EcalEBClusterTriggerPrimitiveDigi::sampleOfInterest() const
{
  if (size_ == 1)
    return 0;
  else if (size_ == 5)
    return 2;
  else
    return -1;
} 

/// get the encoded/compressed Et of interesting sample
int EcalEBClusterTriggerPrimitiveDigi::encodedEt() const 
{
  int sample = sampleOfInterest();
  if (sample != -1)
    return data_[sample].encodedEt();
  else
    return -1;
}
 

int EcalEBClusterTriggerPrimitiveDigi::l1aSpike() const
{
  int sample = sampleOfInterest();
  if (sample != -1)
    return data_[sample].l1aSpike();
  else
    return -1;
}

int EcalEBClusterTriggerPrimitiveDigi::time() const
{
  int sample = sampleOfInterest();
  if (sample != -1)
    return data_[sample].time();
  else
    return -1;
}

bool EcalEBClusterTriggerPrimitiveDigi::isDebug() const
{
  if (size_ == 1)
    return false;
  else if (size_ > 1)
    return true;
  return false;
}

void EcalEBClusterTriggerPrimitiveDigi::setSize(int size) {
  if (size<0) size_=0;
  else if (size>MAXSAMPLES) size_=MAXSAMPLES;
  else size_=size;
}

  
std::ostream& operator<<(std::ostream& s, const EcalEBClusterTriggerPrimitiveDigi& digi) {
  s << digi.id() << " " << digi.size() << " samples " << std::endl;
  for (int i=0; i<digi.size(); i++) 
    s << "  " << digi.sample(i) << std::endl;
  return s;
}

