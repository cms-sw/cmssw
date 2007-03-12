
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

SiStripCluster::SiStripCluster( uint32_t detid, const SiStripDigiRange& range) :
  detId_(detid), firstStrip_(range.first->strip())
{

  amplitudes_.reserve( range.second - range.first);
  int sumx = 0;
  int suma = 0;
  
  uint16_t lastStrip=0;
  bool firstInloop = true;
  for (SiStripDigiIter i=range.first; i!=range.second; i++) {
    
    /// check if digis consecutive
    if (!firstInloop && i->strip() != lastStrip + 1) {
      for (int j=0; j < i->strip()-(lastStrip+1); j++) {
	amplitudes_.push_back( 0);
      }
    }
    lastStrip = i->strip();
    firstInloop = false;
    
    uint16_t amp = i->adc();       // FIXME: gain correction here
    amplitudes_.push_back( amp);
    sumx += i->strip()*amp;
    suma += amp;
  }
  // strip centers are offcet by half pitch w.r.t. strip numbers,
  // so one has to add 0.5 to get the correct barycenter position
  barycenter_ = sumx / static_cast<float>(suma) + 0.5;
  
}

SiStripCluster::SiStripCluster(const uint32_t& detid, 
			       const uint16_t& firstStrip, 
			       std::vector<uint16_t>::const_iterator begin, 
			       std::vector<uint16_t>::const_iterator end) :

  detId_(detid),
  firstStrip_(firstStrip),
  amplitudes_(),
  barycenter_(-1.) 

{
  amplitudes_.resize(end-begin,0);
  copy(begin,end,amplitudes_.begin());
}

float SiStripCluster::barycenter() {

  if (barycenter_ < 0.) {

  int sumx = 0;
  int suma = 0;

  for (uint16_t istrip = 0;istrip < amplitudes_.size();istrip++) {
    sumx += ((firstStrip_+istrip)*amplitudes_[istrip]);
    suma += amplitudes_[istrip];
  }

  // strip centers are offset by half pitch w.r.t. strip numbers,
  // so one has to add 0.5 to get the correct barycenter position
  barycenter_ = sumx / static_cast<float>(suma) + 0.5;
  return barycenter_;
  }

  else return barycenter_;
}
