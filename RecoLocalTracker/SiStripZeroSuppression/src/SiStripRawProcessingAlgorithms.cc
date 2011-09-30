#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingAlgorithms.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingFactory.h"
#include <memory>


SiStripRawProcessingAlgorithms::SiStripRawProcessingAlgorithms(std::auto_ptr<SiStripPedestalsSubtractor> ped,
				 std::auto_ptr<SiStripCommonModeNoiseSubtractor> cmn,
				 std::auto_ptr<SiStripFedZeroSuppression> zs,
				 std::auto_ptr<SiStripAPVRestorer> res,
				 bool doAPVRest,
				 bool useCMMap)
    :  subtractorPed(ped),
       subtractorCMN(cmn),
       suppressor(zs),
       restorer(res),
       doAPVRestore(doAPVRest),
       useCMMeanMap(useCMMap)
    {}


void SiStripRawProcessingAlgorithms::initialize(const edm::EventSetup& es) {
    subtractorPed->init(es);
    subtractorCMN->init(es);
    suppressor->init(es);
    if(restorer.get()) restorer->init(es);
    //if(restorer.get()&&doAPVRestore&&useCMMeanMap) restorer->LoadMeanCMMap(es);
  } 




int16_t SiStripRawProcessingAlgorithms::SuppressVirginRawData(const uint32_t& id, const uint16_t& firstAPV, std::vector<int16_t>& processedRawDigis , edm::DetSet<SiStripDigi>& suppressedDigis ){
     
  //std::vector<int16_t>  processedRawDigisPedSubtracted;
      
      subtractorPed->subtract( id, firstAPV*128,processedRawDigis);
      return this->SuppressProcessedRawData(id, firstAPV, processedRawDigis , suppressedDigis );

      //  if( doAPVRestore ) processedRawDigisPedSubtracted.assign(processedRawDigis.begin(), processedRawDigis.end());
      //subtractorCMN->subtract(id, firstAPV,  processedRawDigis);
      //if( doAPVRestore ) nAPVflagged = algorithms->restorer->InspectAndRestore(id, processedRawDigisPedSubtracted, algorithms->subtractorCMN->getAPVsCM(), processedRawDigis);
      //suppressor->suppress( processedRawDigis, firstAPV,  suppressedDigis );
}

int16_t SiStripRawProcessingAlgorithms::SuppressVirginRawData(const edm::DetSet<SiStripRawDigi>& rawDigis, edm::DetSet<SiStripDigi>& suppressedDigis){
   std::vector<int16_t> RawDigis;
   RawDigis.clear();
   edm::DetSet<SiStripRawDigi>::const_iterator itrawDigis = rawDigis.begin();
   for(; itrawDigis != rawDigis.begin(); ++itrawDigis) RawDigis.push_back(itrawDigis->adc());
   return this->SuppressVirginRawData(rawDigis.id, 0,RawDigis , suppressedDigis);
}
  



int16_t SiStripRawProcessingAlgorithms::SuppressProcessedRawData(const uint32_t& id, const uint16_t& firstAPV, std::vector<int16_t>& processedRawDigis , edm::DetSet<SiStripDigi>& suppressedDigis ){
      std::vector<int16_t>  processedRawDigisPedSubtracted ;
      
      int16_t nAPVFlagged =0;
     // transform(rawDigis->begin(), rawDigis->end(), back_inserter(processedRawDigis), boost::bind(&SiStripRawDigi::adc , _1));
     // std::cout << "here 1 " << id  << " APV " << firstAPV << std::endl;
      if( doAPVRestore ) processedRawDigisPedSubtracted.assign(processedRawDigis.begin(), processedRawDigis.end());
      // std::cout << "here 2" << std::endl;
      subtractorCMN->subtract(id, firstAPV,  processedRawDigis);
      //std::cout << "here 3" << std::endl;
      if( doAPVRestore ) nAPVFlagged = restorer->InspectAndRestore(id, firstAPV, processedRawDigisPedSubtracted, processedRawDigis, subtractorCMN->getAPVsCM() );
      //std::cout << "here 4" << std::endl;
      suppressor->suppress( processedRawDigis, firstAPV,  suppressedDigis ); 
      //std::cout << "here 5" << std::endl;
      return nAPVFlagged;
}


int16_t SiStripRawProcessingAlgorithms::SuppressProcessedRawData(const edm::DetSet<SiStripRawDigi>& rawDigis, edm::DetSet<SiStripDigi>& suppressedDigis){
   std::vector<int16_t> RawDigis;
   RawDigis.clear();
   edm::DetSet<SiStripRawDigi>::const_iterator itrawDigis = rawDigis.begin();
   for(; itrawDigis != rawDigis.begin(); ++itrawDigis) RawDigis.push_back(itrawDigis->adc());
    return this->SuppressProcessedRawData(rawDigis.id, 0, RawDigis , suppressedDigis );
}
