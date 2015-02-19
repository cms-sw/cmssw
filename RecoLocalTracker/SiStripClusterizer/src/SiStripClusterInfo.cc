#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterInfo.h"

#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithmFactory.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithm.h"
#include <cmath>



SiStripClusterInfo::SiStripClusterInfo(const SiStripCluster& cluster,
                                       const edm::EventSetup& setup,
				       const int detId,
				       const std::string & quality)
  : cluster_ptr(&cluster),
    es(setup),
    qualityLabel(quality),
    detId_(detId) {
  es.get<SiStripNoisesRcd>().get(noiseHandle);
  es.get<SiStripGainRcd>().get(gainHandle);
  es.get<SiStripQualityRcd>().get(qualityLabel,qualityHandle);
}

std::pair<uint16_t,uint16_t > SiStripClusterInfo::
chargeLR() const { 
  std::vector<uint8_t>::const_iterator 
    begin( stripCharges().begin() ),
    end( stripCharges().end() ), 
    max; max = max_element(begin,end);
  return std::make_pair( accumulate(begin, max, uint16_t(0) ),
			 accumulate(max+1, end, uint16_t(0) ) );
}


float SiStripClusterInfo::
variance() const {
  float q(0), x1(0), x2(0);
  for(auto 
	begin(stripCharges().begin()), end(stripCharges().end()), it(begin); 
      it!=end; ++it) {
    unsigned i = it-begin;
    q  += (*it);
    x1 += (*it) * (i+0.5);
    x2 += (*it) * (i*i+i+1./3);
  }
  return (x2 - x1*x1/q ) / q;
}

std::vector<float> SiStripClusterInfo::
stripNoisesRescaledByGain() const { 
  SiStripNoises::Range detNoiseRange = noiseHandle->getRange(detId_);
  SiStripApvGain::Range detGainRange = gainHandle->getRange(detId_);

  std::vector<float> results;
  results.reserve(width());
  for(size_t i = 0, e = width(); i < e; i++){
    results.push_back(noiseHandle->getNoise(firstStrip()+i, detNoiseRange) / gainHandle->getStripGain( firstStrip()+i, detGainRange));
  }
  return results;
}

std::vector<float> SiStripClusterInfo::
stripNoises() const {  
  SiStripNoises::Range detNoiseRange = noiseHandle->getRange(detId_);  
  
  std::vector<float> noises;
  noises.reserve(width());
  for(size_t i=0; i < width(); i++){
    noises.push_back( noiseHandle->getNoise( firstStrip()+i, detNoiseRange) );
  }
  return noises;
}

std::vector<float> SiStripClusterInfo::
stripGains() const {  
  SiStripApvGain::Range detGainRange = gainHandle->getRange(detId_);	

  std::vector<float> gains;
  gains.reserve(width());
  for(size_t i=0; i< width(); i++){	
    gains.push_back( gainHandle->getStripGain( firstStrip()+i, detGainRange) );
  } 
  return gains;
}

std::vector<bool> SiStripClusterInfo::
stripQualitiesBad() const {
  std::vector<bool> isBad;
  isBad.reserve(width());
  for(int i=0; i< width(); i++) {
    isBad.push_back( qualityHandle->IsStripBad( detId_, 
						 firstStrip()+i) );
  }
  return isBad;
}

float SiStripClusterInfo::
calculate_noise(const std::vector<float>& noise) const {  
  float noiseSumInQuadrature = 0;
  int numberStripsOverThreshold = 0;
  for(int i=0;i<width();i++) {
    if(stripCharges()[i]!=0) {
      noiseSumInQuadrature += noise.at(i) * noise.at(i);
      numberStripsOverThreshold++;
    }
  }
  return std::sqrt( noiseSumInQuadrature / numberStripsOverThreshold );
} 


bool SiStripClusterInfo::
IsAnythingBad() const {
  std::vector<bool> stripBad = stripQualitiesBad();
  return
    IsApvBad() ||
    IsFiberBad() ||
    IsModuleBad() ||
    accumulate(stripBad.begin(), stripBad.end(), 
	       false,
	       std::logical_or<bool>());
}

bool SiStripClusterInfo::
IsApvBad() const {
  return 
    qualityHandle->IsApvBad( detId_, firstStrip()/128 ) ||
    qualityHandle->IsApvBad( detId_, (firstStrip()+width())/128 ) ;    
}

bool SiStripClusterInfo::
IsFiberBad() const {
  return 
    qualityHandle->IsFiberBad( detId_, firstStrip()/256 ) ||
    qualityHandle->IsFiberBad( detId_, (firstStrip()+width())/256 ) ;
}

bool SiStripClusterInfo::
IsModuleBad() const {
  return qualityHandle->IsModuleBad( detId_ );
}

bool SiStripClusterInfo::
IsModuleUsable() const {
  return qualityHandle->IsModuleUsable( detId_ );
}

std::vector<SiStripCluster> SiStripClusterInfo::
reclusterize(const edm::ParameterSet& conf) const {
  
  std::vector<SiStripCluster> clusters;

  std::vector<uint8_t> charges(stripCharges().begin(),stripCharges().end());
  std::vector<float> gains = stripGains();
  for(unsigned i=0; i < charges.size(); i++)
    charges[i] = (charges[i] < 254) 
      ? static_cast<uint8_t>(charges[i] * gains[i])
      : charges[i];

  std::auto_ptr<StripClusterizerAlgorithm> 
    algorithm = StripClusterizerAlgorithmFactory::create(conf);
  algorithm->initialize(es);

  if( algorithm->stripByStripBegin( detId_ )) {
    for(unsigned i = 0; i<width(); i++)
      algorithm->stripByStripAdd( firstStrip()+i, charges[i], clusters );
    algorithm->stripByStripEnd( clusters );
  }

  return clusters;
}

