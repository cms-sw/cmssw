#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithm.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include <string>

void StripClusterizerAlgorithm::
initialize(const edm::EventSetup& es) {
  uint32_t n_cache_id = es.get<SiStripNoisesRcd>().cacheIdentifier();
  uint32_t g_cache_id = es.get<SiStripGainRcd>().cacheIdentifier();
  uint32_t q_cache_id = es.get<SiStripQualityRcd>().cacheIdentifier();

  if(n_cache_id != noise_cache_id) {
    es.get<SiStripNoisesRcd>().get( noiseHandle );
    noise_cache_id = n_cache_id;
  }
  if(g_cache_id != gain_cache_id) {
    es.get<SiStripGainRcd>().get( gainHandle );
    gain_cache_id = g_cache_id;
  }
  if(q_cache_id != quality_cache_id) {
    es.get<SiStripQualityRcd>().get( qualityLabel, qualityHandle );
    quality_cache_id = q_cache_id;
  }
}


void StripClusterizerAlgorithm::
setDetId(const uint32_t id) {
  gainRange =  gainHandle->getRange(id); 
  noiseRange = noiseHandle->getRange(id);
  qualityRange = qualityHandle->getRange(id);
  detId = id;
}

void StripClusterizerAlgorithm::clusterize(const   edm::DetSetVector<SiStripDigi>& input,  output_t& output) {clusterize_(input, output);}
void StripClusterizerAlgorithm::clusterize(const edmNew::DetSetVector<SiStripDigi>& input, output_t& output) {clusterize_(input, output);}

StripClusterizerAlgorithm::
InvalidChargeException::InvalidChargeException(const SiStripDigi& digi)
  : cms::Exception("Invalid Charge") {
  std::stringstream s;
  s << "Digi charge of " << digi.adc() << " ADC "
    << "is out of range on strip " << digi.strip() << ".  ";
  this->append(s.str());
}
