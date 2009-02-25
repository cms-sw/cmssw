#include "AnalysisDataFormats/SiStripClusterInfo/interface/SiStripClusterInfo.h"

#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/ThreeThresholdStripClusterizer.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"

#include <functional>
#include <cmath>

SiStripClusterInfo::SiStripClusterInfo(const SiStripCluster&  cluster, 
                                       const edm::EventSetup&      es ,
				       std::string qualityLabel)
  : cluster_(&cluster),
    es_(es),
    qualityLabel_(qualityLabel) {
  es.get<SiStripNoisesRcd>().get(noiseHandle_);
  es.get<SiStripGainRcd>().get(gainHandle_);
  es.get<SiStripQualityRcd>().get(qualityLabel_,qualityHandle_);
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

std::vector<float> SiStripClusterInfo::
stripNoisesRescaledByGain() const { 
  std::vector<float> noises = stripNoises();  
  std::vector<float> gains = stripGains();
  transform(noises.begin(), noises.end(), gains.begin(), 
	    noises.begin(), 
	    std::divides<double>());
  return noises;
}

std::vector<float> SiStripClusterInfo::
stripNoises() const {  
  SiStripNoises::Range detNoiseRange = noiseHandle_->getRange(cluster_->geographicalId());  
  
  std::vector<float> noises;
  for(size_t i=0; i< width(); i++){
    noises.push_back( noiseHandle_->getNoise( firstStrip()+i, detNoiseRange) );
  }
  return noises;
}

std::vector<float> SiStripClusterInfo::
stripGains() const {  
  SiStripApvGain::Range detGainRange = gainHandle_->getRange(cluster_->geographicalId());	

  std::vector<float> gains;
  for(size_t i=0; i< width(); i++){	
    gains.push_back( gainHandle_->getStripGain( firstStrip()+i, detGainRange) );
  } 
  return gains;
}

std::vector<bool> SiStripClusterInfo::
stripQualitiesBad() const {
  std::vector<bool> isBad;
  for(int i=0; i< width(); i++) {
    isBad.push_back( qualityHandle_->IsStripBad( cluster_->geographicalId(), 
						 firstStrip()+i) );
  }
  return isBad;
}

float SiStripClusterInfo::
calculate_noise(const std::vector<float>& noise) const {  
  float noiseSumInQuadrature = 0;
  int numberStripsOverThreshold = 0;
  for(int i=0;i<width();i++) {
    if(stripCharges().at(i)!=0) {
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
    qualityHandle_->IsApvBad( cluster_->geographicalId(), firstStrip()/128 ) ||
    qualityHandle_->IsApvBad( cluster_->geographicalId(), (firstStrip()+width())/128 ) ;    
}

bool SiStripClusterInfo::
IsFiberBad() const {
  return 
    qualityHandle_->IsFiberBad( cluster_->geographicalId(), firstStrip()/256 ) ||
    qualityHandle_->IsFiberBad( cluster_->geographicalId(), (firstStrip()+width())/256 ) ;
}

bool SiStripClusterInfo::
IsModuleBad() const {
  return qualityHandle_->IsModuleBad( cluster_->geographicalId() );
}

bool SiStripClusterInfo::
IsModuleUsable() const {
  return qualityHandle_->IsModuleUsable( cluster_->geographicalId() );
}

std::vector<SiStripCluster> SiStripClusterInfo::
reclusterize(float channelThreshold, 
	     float seedThreshold, 
	     float clusterThreshold, 
	     uint8_t maxSequentialHoles, 
	     uint8_t maxSequentialBad,
	     uint8_t maxAdjacentBad) const {
  
  std::vector<uint8_t> charges = stripCharges();
  std::vector<float> gains = stripGains();
  edm::DetSet<SiStripDigi> input_ssd(detId());
  for(int i=0; i<width();i++)
    input_ssd.data.push_back(SiStripDigi( i + firstStrip(), 
					  static_cast<uint16_t>( (charges.at(i)>253)
								 ? charges.at(i)
								 : charges.at(i)*gains.at(i) ) ));
  
  ThreeThresholdStripClusterizer * clusterizer = new ThreeThresholdStripClusterizer(channelThreshold,
										    seedThreshold,
										    clusterThreshold,
										    maxSequentialHoles,
										    maxSequentialBad,
										    maxAdjacentBad);
  edmNew::DetSetVector<SiStripCluster> outputDSV;
  edmNew::DetSetVector<SiStripCluster>::FastFiller output_ssc(outputDSV, detId() );
  clusterizer->init(es_,qualityLabel_);
  clusterizer->clusterizeDetUnit(input_ssd, output_ssc);
  delete clusterizer;

  return outputDSV.data();
}
