#include "RecoLocalTracker/SiStripClusterizer/interface/ThreeThresholdStripClusterizer.h"

#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include <sstream>
#include <cmath>
#include <numeric>
#include <algorithm>

ThreeThresholdStripClusterizer::
ThreeThresholdStripClusterizer(float strip_thr, float seed_thr,float clust_thr, int max_holes, int max_bad, int max_adj) : 
  info(new ESinfo(strip_thr,seed_thr,clust_thr,max_holes,max_bad,max_adj)) {
  ADCs.reserve(128); //largest possible cluster (2APV * 64 adjacent strips per APV)
}


ThreeThresholdStripClusterizer::
~ThreeThresholdStripClusterizer() {
  delete info;
}


void ThreeThresholdStripClusterizer::
init(const edm::EventSetup& es, std::string qualityLabel, std::string thresholdLabel) {
  es.get<SiStripGainRcd>().get(info->gainHandle);
  es.get<SiStripNoisesRcd>().get(info->noiseHandle);
  es.get<SiStripQualityRcd>().get(qualityLabel,info->qualityHandle);
}


inline 
void ThreeThresholdStripClusterizer::
ESinfo::setDetId(uint32_t id) {
  gainRange =  gainHandle->getRange(id); 
  noiseRange = noiseHandle->getRange(id);
  qualityRange = qualityHandle->getRange(id);
}


template<class digiDetSet>
inline
void ThreeThresholdStripClusterizer::
clusterizeDetUnit_(const digiDetSet & digis, output_t& output) {
  if( !info->isModuleUsable( digis.detId() )) return;
  info->setDetId( digis.detId() );
  
  typename digiDetSet::const_iterator   scan(digis.begin()), end(digis.end());
  while( scan != end ) {
    clearCandidate(); 
    while( scan != end  && !candidateEnded( scan->strip() )  )   
      addToCandidate(*scan++);
    if( candidateAccepted() ) {
      transform( ADCs.begin(), ADCs.end(), ADCs.begin(), applyGain(info, firstStrip()) );
      appendBadNeighborsToCandidate();
      output.push_back( SiStripCluster( digis.detId(), firstStrip(), ADCs.begin(), ADCs.end()) );
    }
  }
}


inline 
bool ThreeThresholdStripClusterizer::
candidateEnded(uint16_t testStrip) const {
  uint16_t holes = testStrip - lastStrip - 1;
  return ( !ADCs.empty() &&                         // a candidate exists, and
	   holes > info->MaxSequentialHoles &&      // too many holes if not all are bad strips, and
	   ( holes > info->MaxSequentialBad ||      // (too many bad strips anyway, or 
	     !info->allBadBetween( lastStrip, testStrip ))); // not all holes are bad strips)
}


inline 
void ThreeThresholdStripClusterizer::
addToCandidate(const SiStripDigi& digi) { 
  float noise = info->noise(digi.strip());
  if( !info->bad(digi.strip())            && digi.adc() >= static_cast<uint16_t>( noise * info->ChannelThreshold)) {
    if(!candidateHasSeed) candidateHasSeed = digi.adc() >= static_cast<uint16_t>( noise * info->SeedThreshold);
    if( ADCs.empty() ) lastStrip = digi.strip() - 1; //begin candidate
    while( ++lastStrip < digi.strip() ) ADCs.push_back(0); //pad holes
    ADCs.push_back( digi.adc() );
    noiseSquared += noise*noise;
  }
}


inline 
bool ThreeThresholdStripClusterizer::
candidateAccepted() const {
  return ( candidateHasSeed &&
	   noiseSquared * info->ClusterThresholdSquared
	   <=  std::pow( std::accumulate(ADCs.begin(),ADCs.end(),float(0)), 2));
}


inline 
uint16_t ThreeThresholdStripClusterizer::
applyGain::operator()(uint16_t adc) {
  if(adc > 255) throw InvalidChargeException(SiStripDigi(adc,strip));
  uint16_t charge = static_cast<uint16_t>( adc/info->gain(strip++) + 0.5 ); //adding 0.5 turns truncation into rounding
  return  ( charge > 511 ? 255 : 
	  ( charge > 253 ? 254 : charge ));
}


inline 
void ThreeThresholdStripClusterizer::
appendBadNeighborsToCandidate() {
  uint8_t max = info->MaxAdjacentBad;
  while(0 < max--) {
    if(info->bad(firstStrip()-1)) { ADCs.insert( ADCs.begin(), 0);             }
    if(info->bad( lastStrip + 1)) { ADCs.push_back(0);             lastStrip++;}
  }
}


ThreeThresholdStripClusterizer::
InvalidChargeException::InvalidChargeException(const SiStripDigi& digi)
  : cms::Exception("Invalid Charge") {
  std::stringstream s;
  s << "Digi charge of " << digi.adc() << " ADC "
    << "is out of range on strip " << digi.strip() << ".  "
    << "The ThreeThresholdStripClusterizer algorithm only works "
    << "with input charges less than 256 ADC counts." << std::endl;
  this->append(s.str());
}


void ThreeThresholdStripClusterizer::
clusterizeDetUnit(const edm::DetSet<SiStripDigi>& digis, output_t& output) {
  clusterizeDetUnit_(digis,output);
}

void ThreeThresholdStripClusterizer::
clusterizeDetUnit(const edmNew::DetSet<SiStripDigi>& digis, output_t& output) {
  clusterizeDetUnit_(digis,output);
}
