#include "RecoLocalTracker/SiStripClusterizer/interface/ThreeThresholdStripClusterizer.h"

#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include <functional>
#include <algorithm>
#include <cmath>
#include <sstream>

ThreeThresholdStripClusterizer::
ThreeThresholdStripClusterizer(float strip_thr, float seed_thr,float clust_thr, int max_holes, int max_bad, int max_adj)
  : esinfo(0) {
  thresholds = new thresholdGroup(strip_thr,seed_thr,clust_thr,max_holes,max_bad,max_adj);
}

ThreeThresholdStripClusterizer::
~ThreeThresholdStripClusterizer()   {
  delete thresholds;
  if(esinfo) delete esinfo;
}

void
ThreeThresholdStripClusterizer::
init(const edm::EventSetup& es, std::string qualityLabel, std::string thresholdLabel) {
  if(esinfo) delete esinfo;
  esinfo = new ESinfo(es, qualityLabel);
  amplitudes.reserve(80);
  extDigis.reserve(300);
}

ThreeThresholdStripClusterizer::
ESinfo::ESinfo(const edm::EventSetup& es, std::string qualityLabel)  {
  es.get<SiStripGainRcd>().get(gainHandle);
  es.get<SiStripNoisesRcd>().get(noiseHandle);
  es.get<SiStripQualityRcd>().get(qualityLabel,qualityHandle);
}

void
ThreeThresholdStripClusterizer::
ESinfo::setDetId(uint32_t id) {
  currentDetId = id;
  gainRange =  gainHandle->getRange(id); 
  noiseRange = noiseHandle->getRange(id);
  qualityRange = qualityHandle->getRange(id);
}

template<class digiDetSet>
void 
ThreeThresholdStripClusterizer::
clusterizeDetUnit_(const digiDetSet & digis, edmNew::DetSetVector<SiStripCluster>::FastFiller & output) {
  if( !esinfo->isModuleUsable( digis.detId() )) return;
  esinfo->setDetId( digis.detId() );
  extDigis.clear();  transform( digis.begin(), digis.end(), 
				back_inserter(extDigis), extDigiConstructor(esinfo,thresholds) );

  iter_t left, seed, right = extDigis.begin();
  while( (seed = std::find_if<iter_t>(right,extDigis.end(), isSeed() ))
	 != extDigis.end()  )  {
    right = findClusterEdge<iter_t>( seed, extDigis.end() );
    left  = findClusterEdge<riter_t>( riter_t(seed+1), riter_t(extDigis.begin()) ).base();
    if( aboveClusterThreshold(left,right) )
      clusterize(left,right,output);
  }
}

template<class digiIter>
inline digiIter
ThreeThresholdStripClusterizer::
findClusterEdge(digiIter seed, digiIter end) const {
  digiIter back(seed), test(seed);
  while( !clusterEdgeCondition(back,++test,end) )
    if( test->aboveChannel )
      back = test;
  return back+1;
}
 
template<class digiIter>
inline bool
ThreeThresholdStripClusterizer::
clusterEdgeCondition(digiIter back, digiIter test, digiIter end) const {
  if(test == end) return true;
  uint16_t Nbetween = std::abs( test->strip - back->strip) - 1;
  return ( Nbetween > thresholds->MaxSequentialHoles                            
	   && ( Nbetween > thresholds->MaxSequentialBad                             
		|| esinfo->anyGoodBetween( back->strip,test->strip ) )
	   );
}

inline bool
ThreeThresholdStripClusterizer::
ESinfo::anyGoodBetween(uint16_t a, uint16_t b) const {
  uint16_t strip = 1 + std::min(a,b) ;  
  while( strip < std::max(a,b)  &&  qualityHandle->IsStripBad(qualityRange,strip) )
    ++strip;
  return strip != std::max(a,b);
}

inline bool
ThreeThresholdStripClusterizer::
aboveClusterThreshold(iter_t left, iter_t right) const {
  float charge(0), noise2(0);
  for(iter_t it = left; it < right; it++) {
    if( it->aboveChannel ) {
      noise2 += it->noise * it->noise;
      charge += it->adc;
    }
  }
  return charge*charge  >=  noise2 * thresholds->Cluster * thresholds->Cluster;
}

void
ThreeThresholdStripClusterizer::
clusterize(iter_t left, iter_t right, edmNew::DetSetVector<SiStripCluster>::FastFiller& output) {
  uint8_t preBad  = esinfo->badAdjacent( left->strip,      thresholds->MaxAdjacentBad, -1);
  uint8_t postBad = esinfo->badAdjacent( (right-1)->strip, thresholds->MaxAdjacentBad, +1);
  
  amplitudes.clear();
  amplitudes.resize(preBad,0);
  for(iter_t it = left; it<right; it++) {
    amplitudes.resize( it->strip - left->strip + preBad, 0 ); //pad with 0 any zero-supressed holes
    amplitudes.push_back( it->correctedCharge() );
  }
  amplitudes.resize( postBad + amplitudes.size(), 0);

  output.push_back(SiStripCluster( esinfo->detId(), 
				   left->strip - preBad,
				   amplitudes.begin(),
				   amplitudes.end() ));
}

inline uint8_t
ThreeThresholdStripClusterizer::
ESinfo::badAdjacent(const uint16_t& strip, const uint8_t& maxAdjacentBad, const int8_t direction) const {
  uint8_t count=0;
  while(count < maxAdjacentBad 
	&& qualityHandle->IsStripBad(qualityRange, strip + direction*(1+count) ))
    ++count;
  return count;
}

inline uint16_t 
ThreeThresholdStripClusterizer::
SiStripExtendedDigi::correctedCharge() const { 
  if( !aboveChannel ) return 0;
  if(adc > 255) throw InvalidChargeException(strip,adc);
  uint16_t stripCharge = static_cast<uint16_t>( adc/gain + 0.5 ); //adding 0.5 turns truncation into rounding
  if(stripCharge>511) return 255;
  if(stripCharge>253) return 254;
  return stripCharge;
}

ThreeThresholdStripClusterizer::
InvalidChargeException::InvalidChargeException(uint16_t strip, uint16_t adc)
  : cms::Exception("Invalid Charge") {
  std::stringstream s;
  s << "Digi charge of " << adc << " ADC "
    << "is out of range on strip " << strip << ".  "
    << "The ThreeThresholdStripClusterizer algorithm only works "
    << "with input charges less than 256 ADC counts." << std::endl;
  this->append(s.str());
}



void 
ThreeThresholdStripClusterizer::
clusterizeDetUnit(const edm::DetSet<SiStripDigi> & digis, edmNew::DetSetVector<SiStripCluster>::FastFiller & output) {
  clusterizeDetUnit_(digis,output);
}
void 
ThreeThresholdStripClusterizer::
clusterizeDetUnit(const edmNew::DetSet<SiStripDigi> & digis, edmNew::DetSetVector<SiStripCluster>::FastFiller & output) {
  clusterizeDetUnit_(digis,output);
}
