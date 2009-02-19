#include "RecoLocalTracker/SiStripClusterizer/interface/ThreeThresholdStripClusterizer.h"

#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include "boost/iterator/reverse_iterator.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>

ThreeThresholdStripClusterizer::
ThreeThresholdStripClusterizer(float strip_thr, float seed_thr,float clust_thr, int max_holes, int max_bad, int max_adj)
  : digiInfo(0) {
  thresholds = new thresholdGroup(strip_thr,seed_thr,clust_thr,max_holes,max_bad,max_adj);
}

ThreeThresholdStripClusterizer::
~ThreeThresholdStripClusterizer()   {
  delete thresholds;
  if(digiInfo) delete digiInfo;
  digiInfo=0;
}

void
ThreeThresholdStripClusterizer::
init(const edm::EventSetup& es, std::string qualityLabel, std::string thresholdLabel) {
  if(digiInfo) delete digiInfo;
  digiInfo = new DigiInfo(es, thresholds, qualityLabel);
}

ThreeThresholdStripClusterizer::
DigiInfo::DigiInfo(const edm::EventSetup& es, thresholdGroup* thresholds, std::string qualityLabel) 
  : thresholdHandle(thresholds) {
  es.get<SiStripGainRcd>().get(gainHandle);
  es.get<SiStripNoisesRcd>().get(noiseHandle);
  es.get<SiStripQualityRcd>().get(qualityLabel,qualityHandle);
}

void
ThreeThresholdStripClusterizer::
DigiInfo::setFastAccessDetId(uint32_t id) {
  currentDetId = id;
  gainRange =  gainHandle->getRange(id); 
  noiseRange = noiseHandle->getRange(id);
  qualityRange = qualityHandle->getRange(id);
}

void 
ThreeThresholdStripClusterizer::
clusterizeDetUnit(const edm::DetSet<SiStripDigi> & digis, edmNew::DetSetVector<SiStripCluster>::FastFiller & output) {
  clusterizeDetUnitTemplate(digis,output);
}
void 
ThreeThresholdStripClusterizer::
clusterizeDetUnit(const edmNew::DetSet<SiStripDigi> & digis, edmNew::DetSetVector<SiStripCluster>::FastFiller & output) {
  clusterizeDetUnitTemplate(digis,output);
}

template<class digiDetSet>
void 
ThreeThresholdStripClusterizer::
clusterizeDetUnitTemplate(const digiDetSet & digis, edmNew::DetSetVector<SiStripCluster>::FastFiller & output) {
  if( !digiInfo->isModuleUsable( digis.detId() )) return;
  digiInfo->setFastAccessDetId( digis.detId() );

  typedef typename digiDetSet::const_iterator iter_t;
  typedef boost::reverse_iterator<iter_t> riter_t;

  iter_t left, seed, right = digis.begin();
  while( (seed = std::find_if(right,digis.end(),isSeed(digiInfo)))
	 != digis.end()  )  {
    right = findClusterEdge( seed, digis.end() );
    left  = findClusterEdge( riter_t(seed+1), riter_t(digis.begin()) ).base();
    if( aboveClusterThreshold(left,right) )
      clusterize(left,right,output);
  }
}

inline bool
ThreeThresholdStripClusterizer::
isSeed::operator()(const SiStripDigi& digi) {
  return digiInfo->isAboveSeed(digi) && digiInfo->isGoodStrip(digi) ; 
}

template<class digiIter>
inline digiIter
ThreeThresholdStripClusterizer::
findClusterEdge(digiIter seed, digiIter end) const {
  digiIter back(seed), test(seed);
  while( !clusterEdgeCondition(back,++test,end) )
    if(digiInfo->includeInCluster(*test))
      back = test;
  return back+1;
}
 
template<class digiIter>
inline bool
ThreeThresholdStripClusterizer::
clusterEdgeCondition(digiIter back, digiIter test, digiIter end) const {
  if(test == end) return true;
  uint16_t Nbetween = std::abs( test->strip() - back->strip()) - 1;
  return ( Nbetween > digiInfo->maxSequentialHoles()                                
	   && ( Nbetween > digiInfo->maxSequentialBad()                             
		|| digiInfo->anyGoodBetween( back->strip(),test->strip() ) )
	   );
}

inline bool
ThreeThresholdStripClusterizer::DigiInfo::
anyGoodBetween(uint16_t a, uint16_t b) const {
  uint16_t strip = 1 + std::min(a,b) ;  
  while( strip < std::max(a,b)  &&  qualityHandle->IsStripBad(qualityRange,strip) )
    ++strip;
  return strip != std::max(a,b);
}

template<class digiIter>
inline bool
ThreeThresholdStripClusterizer::
aboveClusterThreshold(digiIter left, digiIter right) const {
  float charge(0), noise2(0);
  for(digiIter it = left; it < right; it++) {
    if( digiInfo->includeInCluster(*it) ) {
      noise2 += std::pow(digiInfo->noise(*it), 2);
      charge += it->adc();
    }
  }
  return charge*charge  >=  noise2 * std::pow(digiInfo->clusterThreshold(), 2);
}

template<class digiIter>
void
ThreeThresholdStripClusterizer::
clusterize(digiIter left, digiIter right, edmNew::DetSetVector<SiStripCluster>::FastFiller& output) {
  uint8_t preBad  = digiInfo->nBadBeforeUpToMaxAdjacent(*left);
  uint8_t postBad = digiInfo->nBadAfterUpToMaxAdjacent(*(right-1));
  
  amplitudes.clear();
  amplitudes.resize(preBad,0);
  for(digiIter it = left; it<right; it++) {
    amplitudes.resize( it->strip() - left->strip() + preBad, 0 ); //pad with 0 any zero-supressed holes
    amplitudes.push_back( digiInfo->correctedCharge(*it) );
  }
  amplitudes.resize( postBad + amplitudes.size(), 0);

  output.push_back(SiStripCluster( digiInfo->detId(), 
				   left->strip() - preBad,
				   amplitudes.begin(),
				   amplitudes.end() ));
}

inline uint8_t
ThreeThresholdStripClusterizer::
DigiInfo::nBadBeforeUpToMaxAdjacent(const SiStripDigi& digi) const {
  uint8_t count=0;
  while(count < maxAdjacentBad() && qualityHandle->IsStripBad(qualityRange, digi.strip()-1-count ))
    ++count;
  return count;
}

inline uint8_t
ThreeThresholdStripClusterizer::
DigiInfo::nBadAfterUpToMaxAdjacent(const SiStripDigi& digi) const {
  uint8_t count=0;
  while(count < maxAdjacentBad() && qualityHandle->IsStripBad(qualityRange, digi.strip()+1+count ))
    ++count;
  return count;
}

inline uint16_t 
ThreeThresholdStripClusterizer::
DigiInfo::correctedCharge(const SiStripDigi& digi) const { 
  if(!includeInCluster(digi)) return 0;
  if(digi.adc() > 255) throw InvalidChargeException(digi);
  uint16_t stripCharge = static_cast<uint16_t>( digi.adc()/gain(digi) + 0.5 ); //adding 0.5 turns truncation into rounding
  if(stripCharge>511) return 255;
  if(stripCharge>253) return 254;
  return stripCharge;
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
