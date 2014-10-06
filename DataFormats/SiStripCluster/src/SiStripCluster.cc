
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

SiStripCluster::SiStripCluster(const SiStripDigiRange& range) :
  firstStrip_(range.first->strip()),
  error_x(-99999.9)
{

  uint16_t lastStrip=0;
  bool firstInloop = true;
  size_=0;
  assert( ((range.second-1)->strip()-range.first->strip()) < MAX_SIZE);
  for (SiStripDigiIter i=range.first; i!=range.second; i++) {
    
    /// check if digis consecutive
    if (!firstInloop && i->strip() != lastStrip + 1) {
      for (int j=0; j < i->strip()-(lastStrip+1); j++) {
	amplitudes_[size_++] = 0;
      }
    }
    lastStrip = i->strip();
    firstInloop = false;
    
    amplitudes_[size_++] = i->adc(); 
  }
}


float SiStripCluster::barycenter() const{
  int sumx = 0;
  int suma = 0;
  size_t asize = size();
  for (size_t i=0;i<asize;++i) {
    sumx += (firstStrip_+i)*(amplitudes_[i]);
    suma += amplitudes_[i];
  }
  
  // strip centers are offcet by half pitch w.r.t. strip numbers,
  // so one has to add 0.5 to get the correct barycenter position
  return sumx / static_cast<float>(suma) + 0.5f;
}
