#include "DataFormats/EcalDigi/interface/EcalPseudoStripInputDigi.h"


EcalPseudoStripInputDigi::EcalPseudoStripInputDigi() : size_(0), data_(MAXSAMPLES) {
}

EcalPseudoStripInputDigi::EcalPseudoStripInputDigi(const EcalTriggerElectronicsId& id) : id_(id),
										   size_(0), data_(MAXSAMPLES) {
}

int EcalPseudoStripInputDigi::sampleOfInterest() const
{
  if (size_ == 1)
    return 0;
  else if (size_ == 5)
    return 2;
  else
    return -1;
} 

/// get the pseudoStrip input of interesting sample
int EcalPseudoStripInputDigi::pseudoStripInput() const 
{
  int sample = sampleOfInterest();
  if (sample != -1)
    return data_[sample].pseudoStripInput();
  else
    return -1;
}
  
/// get the fine-grain bit of interesting sample
bool EcalPseudoStripInputDigi::fineGrain() const 
{ 
  int sample = sampleOfInterest();
  if (sample != -1)
    return data_[sample].fineGrain();
  else
    return false;
}

bool EcalPseudoStripInputDigi::isDebug() const
{
  if (size_ == 1)
    return false;
  else if (size_ > 1)
    return true;
  return false;
}

void EcalPseudoStripInputDigi::setSize(int size) {
  if (size<0) size_=0;
  else if (size>MAXSAMPLES) size_=MAXSAMPLES;
  else size_=size;
}

  
std::ostream& operator<<(std::ostream& s, const EcalPseudoStripInputDigi& digi) {
  s << digi.id() << " " << digi.size() << " samples " << std::endl;
  for (int i=0; i<digi.size(); i++) 
    s << "  " << digi.sample(i) << std::endl;
  return s;
}

