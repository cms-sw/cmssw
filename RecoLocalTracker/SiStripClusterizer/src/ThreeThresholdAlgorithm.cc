#include "RecoLocalTracker/SiStripClusterizer/interface/ThreeThresholdAlgorithm.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include <cmath>
#include <numeric>

ThreeThresholdAlgorithm::
ThreeThresholdAlgorithm(float chan, float seed, float cluster, unsigned holes, unsigned bad, unsigned adj, std::string qL) 
  : ChannelThreshold( chan ), SeedThreshold( seed ), ClusterThresholdSquared( cluster*cluster ),
    MaxSequentialHoles( holes ), MaxSequentialBad( bad ), MaxAdjacentBad( adj ) {
  qualityLabel = (qL);
  ADCs.reserve(128);
}

template<class digiDetSet>
inline
void ThreeThresholdAlgorithm::
clusterizeDetUnit_(const digiDetSet& digis, output_t::FastFiller& output) {
  if( !isModuleUsable( digis.detId() )) return;
  setDetId( digis.detId() );
  
  typename digiDetSet::const_iterator  
    scan( digis.begin() ), 
    end(  digis.end() );

  clearCandidate();
  while( scan != end ) {
    while( scan != end  && !candidateEnded( scan->strip() ) ) 
      addToCandidate(*scan++);
    endCandidate(output);
  }
}

inline 
bool ThreeThresholdAlgorithm::
candidateEnded(const uint16_t& testStrip) const {
  uint16_t holes = testStrip - lastStrip - 1;
  return ( !ADCs.empty() &&                    // a candidate exists, and
	   holes > MaxSequentialHoles &&       // too many holes if not all are bad strips, and
	   ( holes > MaxSequentialBad ||       // (too many bad strips anyway, or 
	     !allBadBetween( lastStrip, testStrip ))); // not all holes are bad strips)
}

inline 
void ThreeThresholdAlgorithm::
addToCandidate(const SiStripDigi& digi) { 
  float Noise = noise( digi.strip() );
  if( bad(digi.strip()) || digi.adc() < static_cast<uint16_t>( Noise * ChannelThreshold))
    return;

  if(candidateLacksSeed) candidateLacksSeed  =  digi.adc() < static_cast<uint16_t>( Noise * SeedThreshold);
  if(ADCs.empty()) lastStrip = digi.strip() - 1; // begin candidate
  while( ++lastStrip < digi.strip() ) ADCs.push_back(0); // pad holes

  ADCs.push_back( digi.adc() );
  noiseSquared += Noise*Noise;
}

template <class T>
inline
void ThreeThresholdAlgorithm::
endCandidate(T& out) {
  if(candidateAccepted()) {
    applyGains();
    appendBadNeighbors();
    out.push_back(SiStripCluster(currentId(), firstStrip(), ADCs.begin(), ADCs.end()));
  }
  clearCandidate();  
}

inline 
bool ThreeThresholdAlgorithm::
candidateAccepted() const {
  return ( !candidateLacksSeed &&
	   noiseSquared * ClusterThresholdSquared
	   <=  std::pow( std::accumulate(ADCs.begin(),ADCs.end(),float(0)), 2));
}

inline
void ThreeThresholdAlgorithm::
applyGains() {
  uint16_t strip = firstStrip();
  for( std::vector<uint16_t>::iterator adc = ADCs.begin();  adc != ADCs.end();  adc++) {
    if(*adc > 255) throw InvalidChargeException( SiStripDigi(strip,*adc) );
    if(*adc > 253) continue; //saturated, do not scale
    uint16_t charge = static_cast<uint16_t>( *adc/gain(strip++) + 0.5 ); //adding 0.5 turns truncation into rounding
    *adc = ( charge > 1022 ? 255 : 
           ( charge >  253 ? 254 : charge ));
  }
}

inline 
void ThreeThresholdAlgorithm::
appendBadNeighbors() {
  uint8_t max = MaxAdjacentBad;
  while(0 < max--) {
    if( bad( firstStrip()-1) ) { ADCs.insert( ADCs.begin(), 0);  }
    if( bad(  lastStrip + 1) ) { ADCs.push_back(0); lastStrip++; }
  }
}


void ThreeThresholdAlgorithm::clusterizeDetUnit(const    edm::DetSet<SiStripDigi>& digis, output_t::FastFiller& output) {clusterizeDetUnit_(digis,output);}
void ThreeThresholdAlgorithm::clusterizeDetUnit(const edmNew::DetSet<SiStripDigi>& digis, output_t::FastFiller& output) {clusterizeDetUnit_(digis,output);}

inline
bool ThreeThresholdAlgorithm::
stripByStripBegin(uint32_t id) {
  if( !isModuleUsable( id )) return false;
  setDetId( id );
  clearCandidate();
  return true;
}

inline
void ThreeThresholdAlgorithm::
stripByStripAdd(uint16_t strip, uint16_t adc, std::vector<SiStripCluster>& out) {
  if(candidateEnded(strip))
    endCandidate(out);
  addToCandidate(SiStripDigi(strip,adc));
}

inline
void ThreeThresholdAlgorithm::
stripByStripEnd(std::vector<SiStripCluster>& out) { 
  endCandidate(out);
}
