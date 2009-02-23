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
  adc.reserve(128); //largest possible cluster after median common mode noise subtraction (64+64 over 2 apv)
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
    clear(); 
    while( scan != end  && !edgeCondition( scan->strip() )  )   
      record(*scan++);
    if( foundCluster() ) {
      transform( adc.begin(), adc.end(), adc.begin(), applyGain(info, firstStrip()) );
      appendBadNeighbors();
      output.push_back( SiStripCluster( digis.detId(), firstStrip(), adc.begin(), adc.end()) );
    }
  }
}

inline 
bool ThreeThresholdStripClusterizer::
edgeCondition(uint16_t testStrip) const {
  uint16_t Nbetween = testStrip - lastStrip - 1;
  return ( !adc.empty()                                                  //  exists a current cluster
	   && ( Nbetween > info->MaxSequentialHoles                      //  AND too many holes
		&& ( Nbetween > info->MaxSequentialBad                   //      AND ( too many bad holes
		     || info->anyGoodBetween( lastStrip, testStrip )))); //             OR  not all holes bad )
}

inline 
void ThreeThresholdStripClusterizer::
record(const SiStripDigi& digi) { 
  float noise = info->noise(digi.strip());
  if( !info->bad(digi.strip()) && digi.adc() >= static_cast<uint16_t>( noise * info->ChannelThreshold)) {
    foundSeed = foundSeed      || digi.adc() >= static_cast<uint16_t>( noise * info->SeedThreshold);
    noiseSquared += noise*noise;
    if( adc.empty() ) lastStrip = digi.strip() - 1;
    while( ++lastStrip < digi.strip() ) adc.push_back(0); //pad holes
    adc.push_back(digi.adc());
  }
}

inline 
bool ThreeThresholdStripClusterizer::
foundCluster() const {
  return ( foundSeed &&
	   noiseSquared * info->ClusterThresholdSquared
	   <=  std::pow( std::accumulate(adc.begin(),adc.end(),float(0)), 2));
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
appendBadNeighbors() {
  uint8_t max = info->MaxAdjacentBad;
  while(0 < max--) {
    if(info->bad(firstStrip()-1)) { reverse(adc.begin(),adc.end()); adc.push_back(0); reverse(adc.begin(),adc.end()); }
    if(info->bad( lastStrip + 1)) {                                 adc.push_back(0); lastStrip++;}
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
