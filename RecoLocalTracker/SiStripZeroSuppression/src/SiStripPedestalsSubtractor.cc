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

void SiStripPedestalsSubtractor::subtract(const uint32_t& id, const uint16_t& firstStrip, std::vector<int16_t>& digis) {subtract_(id, firstStrip, digis, digis);}
void SiStripPedestalsSubtractor::subtract(const edm::DetSet<SiStripRawDigi>& input, std::vector<int16_t>& output) {subtract_(input.id, 0, input, output);}

template <class input_t>
inline
void SiStripPedestalsSubtractor::
subtract_(const uint32_t& id, const uint16_t& firstStrip, const input_t& input, std::vector<int16_t>& output) {
  try {

    pedestals.resize(firstStrip + input.size());
    SiStripPedestals::Range pedestalsRange = pedestalsHandle->getRange(id);
    pedestalsHandle->allPeds(pedestals, pedestalsRange);

    typename input_t::const_iterator inDigi = input.begin();
    std::vector<int>::const_iterator ped = pedestals.begin() + firstStrip;  
    std::vector<int16_t>::iterator   outDigi = output.begin();

    while( inDigi != input.end() ) {
      
      *outDigi = ( *ped > 895 )        
	? eval(*inDigi) - *ped + 1024
	: eval(*inDigi) - *ped;
      
      if(fedmode_ && *outDigi < 0) //FED bottoms out at 0
	*outDigi=0;
      
      ++inDigi; 
      ++ped; 
      ++outDigi;
    }


  } catch(cms::Exception& e){
    edm::LogError("SiStripPedestalsSubtractor")  
      << "[SiStripPedestalsSubtractor::subtract] DetId " << id << " propagating error from SiStripPedestal" << e.what();
    output.clear();
  }

}
