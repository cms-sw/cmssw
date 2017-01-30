#include "DataFormats/EcalDigi/interface/EcalEBTriggerPrimitiveDigi.h"
#include <iostream>


EcalEBTriggerPrimitiveDigi::EcalEBTriggerPrimitiveDigi() : size_(0), data_(MAXSAMPLES) {
}
//EcalTriggerPrimitiveDigi::EcalTriggerPrimitiveDigi(const EcalTrigTowerDetId& id) : id_(id),
//size_(0), data_(MAXSAMPLES) {
//}

EcalEBTriggerPrimitiveDigi::EcalEBTriggerPrimitiveDigi(const EBDetId& id) : id_(id),
										   size_(0), data_(MAXSAMPLES) {
}

void EcalEBTriggerPrimitiveDigi::setSample(int i, const EcalTriggerPrimitiveSample& sam) 
{
//  std::cout << " In setSample  i " << i << "  sam " << sam << std::endl;  
  data_[i]=sam;
//  std::cout << " In setSample data_[i] " << data_[i] << std::endl;  
  
}

int EcalEBTriggerPrimitiveDigi::sampleOfInterest() const
{
  if (size_ == 1)
    return 0;
  else if (size_ == 5)
    return 2;
  else
    return -1;
} 

/// get the encoded/compressed Et of interesting sample
int EcalEBTriggerPrimitiveDigi::compressedEt() const 
{
  int sample = sampleOfInterest();
  if (sample != -1)
    return data_[sample].compressedEt();
  else
    return -1;
}
  
/// get the fine-grain bit of interesting sample
bool EcalEBTriggerPrimitiveDigi::fineGrain() const 
{ 
  int sample = sampleOfInterest();
  if (sample != -1)
    return data_[sample].fineGrain();
  else
    return false;
}
/// get the Trigger tower Flag of interesting sample
int EcalEBTriggerPrimitiveDigi::ttFlag() const 
{ 
  int sample = sampleOfInterest();
  if (sample != -1)
    return data_[sample].ttFlag();
  else
    return -1;
} 

int EcalEBTriggerPrimitiveDigi::sFGVB() const
{
  int sample = sampleOfInterest();
  if (sample != -1)
    return data_[sample].l1aSpike();
  else
    return -1;
}

bool EcalEBTriggerPrimitiveDigi::isDebug() const
{
  if (size_ == 1)
    return false;
  else if (size_ > 1)
    return true;
  return false;
}

void EcalEBTriggerPrimitiveDigi::setSize(int size) {
  if (size<0) size_=0;
  else if (size>MAXSAMPLES) size_=MAXSAMPLES;
  else size_=size;
}

  
std::ostream& operator<<(std::ostream& s, const EcalEBTriggerPrimitiveDigi& digi) {
  s << digi.id() << " " << digi.size() << " samples " << std::endl;
  for (int i=0; i<digi.size(); i++) 
    s << "  " << digi.sample(i) << std::endl;
  return s;
}

