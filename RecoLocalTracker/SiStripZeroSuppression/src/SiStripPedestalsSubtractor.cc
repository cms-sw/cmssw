#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripPedestalsSubtractor.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include "FWCore/Utilities/interface/Exception.h"

void SiStripPedestalsSubtractor::init(const edm::EventSetup& es){
  uint32_t p_cache_id = es.get<SiStripPedestalsRcd>().cacheIdentifier();
  if(p_cache_id != peds_cache_id) {
    es.get<SiStripPedestalsRcd>().get(pedestalsHandle);
    peds_cache_id = p_cache_id;
  }
}


void SiStripPedestalsSubtractor::subtract(const edm::DetSet<SiStripRawDigi>& input, std::vector<int16_t>& output){
  try {

    pedestals.resize(input.size());
    SiStripPedestals::Range pedestalsRange = pedestalsHandle->getRange(input.id);
    pedestalsHandle->allPeds(pedestals, pedestalsRange);

    edm::DetSet<SiStripRawDigi>::const_iterator 
      inDigi = input.begin();

    std::vector<int>::const_iterator              
      ped = pedestals.begin();  

    std::vector<int16_t>::iterator            
      outDigi = output.begin();

    while( inDigi != input.end() ) {

      *outDigi = ( *ped > 895 ) 
	? inDigi->adc() - *ped + 1024
	: inDigi->adc() - *ped;

      ++inDigi; 
      ++ped; 
      ++outDigi;
    }


  } catch(cms::Exception& e){
    edm::LogError("SiStripPedestalsSubtractor")  
      << "[SiStripPedestalsSubtractor::subtract] DetId " << input.id << " propagating error from SiStripPedestal" << e.what();
    output.clear();
  }

}
