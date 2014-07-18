#include "RecoLocalTracker/SiStripClusterizer/interface/ThreeThresholdAlgorithm.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include <cmath>
#include <numeric>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

ThreeThresholdAlgorithm::
ThreeThresholdAlgorithm(float chan, float seed, float cluster, unsigned holes, unsigned bad, unsigned adj, std::string qL, 
			bool setDetId, bool removeApvShots) 
  : ChannelThreshold( chan ), SeedThreshold( seed ), ClusterThresholdSquared( cluster*cluster ),
    MaxSequentialHoles( holes ), MaxSequentialBad( bad ), MaxAdjacentBad( adj ), RemoveApvShots(removeApvShots) {
  _setDetId=setDetId;
  qualityLabel = (qL);
  ADCs.reserve(128);
}

template<class digiDetSet>
inline
void ThreeThresholdAlgorithm::
clusterizeDetUnit_(const digiDetSet& digis, output_t::FastFiller& output) {
  if(isModuleBad(digis.detId())) return;
  if (!setDetId( digis.detId() )) return;

#ifdef EDM_ML_DEBUG
  if(!isModuleUsable(digis.detId() )) 
    LogWarning("ThreeThresholdAlgorithm") << " id " << digis.detId() << " not usable???" << std::endl;
#endif

  
  typename digiDetSet::const_iterator  
    scan( digis.begin() ), 
    end(  digis.end() );

  if(RemoveApvShots){
    ApvCleaner.clean(digis,scan,end);
  }

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
  return ( ( (!ADCs.empty())  &                    // a candidate exists, and
	     (holes > MaxSequentialHoles )       // too many holes if not all are bad strips, and
	     ) && 
	   ( holes > MaxSequentialBad ||       // (too many bad strips anyway, or 
	     !allBadBetween( lastStrip, testStrip ) // not all holes are bad strips)
	     )
	   );
}

inline 
void ThreeThresholdAlgorithm::
addToCandidate(uint16_t strip, uint8_t adc) { 
  float Noise = noise( strip );
  if(  adc < static_cast<uint8_t>( Noise * ChannelThreshold) || bad(strip) )
    return;

  if(candidateLacksSeed) candidateLacksSeed  =  adc < static_cast<uint8_t>( Noise * SeedThreshold);
  if(ADCs.empty()) lastStrip = strip - 1; // begin candidate
  while( ++lastStrip < strip ) ADCs.push_back(0); // pad holes

  ADCs.push_back( adc );
  noiseSquared += Noise*Noise;
}

template <class T>
inline
void ThreeThresholdAlgorithm::
endCandidate(T& out) {
  if(candidateAccepted()) {
    applyGains();
    appendBadNeighbors();
    out.push_back(SiStripCluster(firstStrip(), ADCs.begin(), ADCs.end()));
  }
  clearCandidate();  
}

inline 
bool ThreeThresholdAlgorithm::
candidateAccepted() const {
  return ( !candidateLacksSeed &&
	   noiseSquared * ClusterThresholdSquared
	   <=  std::pow( float(std::accumulate(ADCs.begin(),ADCs.end(), int(0))), 2.f));
}

inline
void ThreeThresholdAlgorithm::
applyGains() {
  uint16_t strip = firstStrip();
  for( auto &  adc :  ADCs) {
#ifdef EDM_ML_DEBUG
    if(adc > 255) throw InvalidChargeException( SiStripDigi(strip,adc) );
#endif
    // if(adc > 253) continue; //saturated, do not scale
    auto charge = int( float(adc)/gain(strip++) + 0.5f ); //adding 0.5 turns truncation into rounding
    if(adc < 254) adc = ( charge > 1022 ? 255 : 
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
  if (!setDetId( id )) return false;
#ifdef EDM_ML_DEBUG
  assert(isModuleUsable( id ));
#endif
  clearCandidate();
  return true;
}

inline
void ThreeThresholdAlgorithm::
stripByStripAdd(uint16_t strip, uint8_t adc, std::vector<SiStripCluster>& out) {
  if(candidateEnded(strip)) endCandidate(out);
  addToCandidate(SiStripDigi(strip,adc));
}

inline
void ThreeThresholdAlgorithm::
stripByStripEnd(std::vector<SiStripCluster>& out) { 
  endCandidate(out);
}
