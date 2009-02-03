#include "AnalysisDataFormats/SiStripClusterInfo/interface/SiStripClusterInfo.h"

#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include <functional>
#include <cmath>

SiStripClusterInfo::SiStripClusterInfo(const SiStripCluster&  cluster, 
                                       const edm::EventSetup&      es )
  : cluster_(&cluster),
    es_(es) {
  es.get<SiStripNoisesRcd>().get(noiseHandle_);
  es.get<SiStripGainRcd>().get(gainHandle_);
  es.get<SiStripQualityRcd>().get(qualityHandle_);
}

std::pair<uint16_t,uint16_t > 
SiStripClusterInfo::chargeLR() const { 
  std::vector<uint8_t>::const_iterator 
    begin, end, max;
  begin = stripCharges().begin();
  end   = stripCharges().end();
  max = max_element(begin,end); 
  
  return std::make_pair( accumulate(begin, max, uint16_t(0) ),
			 accumulate(max+1, end, uint16_t(0) ) );
}

std::vector<float> 
SiStripClusterInfo::stripNoisesRescaledByGain() const { 
  std::vector<float> noises = stripNoises();  
  std::vector<float> gains = stripGains();
  transform(noises.begin(), noises.end(), gains.begin(), 
	    noises.begin(), 
	    std::divides<double>());
  return noises;
}

std::vector<float> 
SiStripClusterInfo::stripNoises() const {  
  SiStripNoises::Range detNoiseRange = noiseHandle_->getRange(cluster_->geographicalId());  
  
  std::vector<float> noises;
  for(size_t i=0; i< width(); i++){
    noises.push_back( noiseHandle_->getNoise(firstStrip()+i,detNoiseRange) );
  }
  return noises;
}

std::vector<float> 
SiStripClusterInfo::stripGains() const {  
  SiStripApvGain::Range detGainRange = gainHandle_->getRange(cluster_->geographicalId());	

  std::vector<float> gains;
  for(size_t i=0; i< width(); i++){	
    gains.push_back( gainHandle_->getStripGain(firstStrip()+i,detGainRange) );
  } 
  return gains;
}

std::vector<bool>
SiStripClusterInfo::stripQualitiesBad() const {
  std::vector<bool> isBad;
  for(int i=0; i< width(); i++) {
    isBad.push_back( qualityHandle_->IsStripBad( cluster_->geographicalId(), 
						firstStrip() + i) );
  }
  return isBad;
}

float
SiStripClusterInfo::calculate_noise(const std::vector<float>& n) const {  
  float noiseSumInQuadrature = inner_product( n.begin(), n.end(), n.begin(), 0.0);
  float numberStripsOverThreshold = count_if( stripCharges().begin(), stripCharges().end(), std::bind1st( std::not_equal_to<uint8_t>(), 0 )  );
  return std::sqrt( noiseSumInQuadrature / numberStripsOverThreshold );
} 


bool
SiStripClusterInfo::IsAnythingBad() const {
  std::vector<bool> stripBad = stripQualitiesBad();
  return
    IsApvBad() ||
    IsFiberBad() ||
    IsModuleBad() ||
    accumulate(stripBad.begin(), stripBad.end(), 
	       false,
	       std::logical_or<bool>());
}

bool
SiStripClusterInfo::IsApvBad() const {
  return 
    qualityHandle_->IsApvBad( cluster_->geographicalId(), firstStrip()/128 ) ||
    qualityHandle_->IsApvBad( cluster_->geographicalId(), (firstStrip()+width())/128 ) ;    
}

bool
SiStripClusterInfo::IsFiberBad() const {
  return 
    qualityHandle_->IsFiberBad( cluster_->geographicalId(), firstStrip()/256 ) ||
    qualityHandle_->IsFiberBad( cluster_->geographicalId(), (firstStrip()+width())/256 ) ;
}

bool
SiStripClusterInfo::IsModuleBad() const {
  return qualityHandle_->IsModuleBad( cluster_->geographicalId() );
}

bool
SiStripClusterInfo::IsModuleUsable() const {
  return qualityHandle_->IsModuleUsable( cluster_->geographicalId() );
}

