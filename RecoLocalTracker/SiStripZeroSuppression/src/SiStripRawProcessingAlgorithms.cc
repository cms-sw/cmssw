#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingAlgorithms.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingFactory.h"
#include <memory>


SiStripRawProcessingAlgorithms::SiStripRawProcessingAlgorithms(
    std::unique_ptr<SiStripPedestalsSubtractor> ped,
    std::unique_ptr<SiStripCommonModeNoiseSubtractor> cmn,
    std::unique_ptr<SiStripFedZeroSuppression> zs,
    std::unique_ptr<SiStripAPVRestorer> res,
    bool doAPVRest, bool useCMMap)
: subtractorPed(std::move(ped)),
  subtractorCMN(std::move(cmn)),
  suppressor(std::move(zs)),
  restorer(std::move(res)),
  doAPVRestore(doAPVRest),
  useCMMeanMap(useCMMap)
{}


void SiStripRawProcessingAlgorithms::initialize(const edm::EventSetup& es) {
    subtractorPed->init(es);
    subtractorCMN->init(es);
    suppressor->init(es);
    if(restorer.get()) restorer->init(es);
    
    edm::ESHandle<TrackerGeometry> tracker; 
    es.get<TrackerDigiGeometryRecord>().get( tracker );
    const TrackerGeometry &trGeotmp(*tracker);
    trGeo = &trGeotmp;
}
void SiStripRawProcessingAlgorithms::initialize(const edm::EventSetup& es, const edm::Event& e){
  this->initialize(es);
  if(restorer.get()&&doAPVRestore&&useCMMeanMap) restorer->LoadMeanCMMap(e);
  
}


//Suppressors Hybrid
//--------------------------------------------------------

uint16_t SiStripRawProcessingAlgorithms::SuppressHybridData(const edm::DetSet<SiStripDigi>& inDigis, edm::DetSet<SiStripDigi>& suppressedDigis, std::vector<int16_t>& RawDigis){
     
     uint32_t id = inDigis.id;
     this->ConvertHybridDigiToRawDigiVector(id, inDigis, RawDigis);
  
     return this->SuppressHybridData(id, 0, RawDigis , suppressedDigis );
}


//IMPORTANT: don't forget the conversion from  hybrids on the bad APVs (*2 -1024)
uint16_t SiStripRawProcessingAlgorithms::SuppressHybridData(const uint32_t& id, const uint16_t& firstAPV, std::vector<int16_t>& processedRawDigis, edm::DetSet<SiStripDigi>& suppressedDigis){
	  std::vector<int16_t>  processedRawDigisPedSubtracted ;
      processedRawDigisPedSubtracted.clear();
      int16_t nAPVFlagged =0;
     
     processedRawDigisPedSubtracted.assign(processedRawDigis.begin(), processedRawDigis.end());
     subtractorCMN->subtract(id, firstAPV,  processedRawDigis);
    
     nAPVFlagged = restorer->InspectAndRestore(id, firstAPV, processedRawDigisPedSubtracted, processedRawDigis, subtractorCMN->getAPVsCM() );
     
     
     
      const std::vector<bool>& apvf = this->GetAPVFlags();
      for(uint16_t APV=firstAPV ; APV< processedRawDigis.size()/128 + firstAPV; ++APV){
  		 
      	if(apvf[APV]){
      	    std::vector<int16_t> singleAPVdigi;
  			singleAPVdigi.clear();
      		singleAPVdigi.assign(processedRawDigis[(APV-firstAPV)*128], processedRawDigis[(APV-firstAPV+1)*128-1]);  
      		//for(int16_t strip = (APV-firstAPV)*128; strip < (APV-firstAPV+1)*128; ++strip) singleAPVdigi.push_back(processedRawDigis[strip]);     		
      		suppressor->suppress(singleAPVdigi, APV,  suppressedDigis );
      	}else{
      		for(int16_t strip = (APV-firstAPV)*128; strip < (APV-firstAPV+1)*128; ++strip){
        		if(processedRawDigisPedSubtracted[strip] > 0) suppressedDigis.push_back(SiStripDigi(strip+firstAPV*128, processedRawDigisPedSubtracted[strip] ));
        		//suppressedDigis.push_back(SiStripDigi(strip+firstAPV*128, ( processedRawDigisPedSubtracted[strip] < 0 ? 0 : suppressor->truncate(processedRawDigisPedSubtracted[strip])))); 
      		}
      	}
      }	
      	  	
     return nAPVFlagged;
}


void SiStripRawProcessingAlgorithms::ConvertHybridDigiToRawDigiVector(const uint32_t& id, const edm::DetSet<SiStripDigi>& inDigis, std::vector<int16_t>& RawDigis){
     
	 const StripGeomDetUnit* StripModuleGeom =(const StripGeomDetUnit*)trGeo->idToDetUnit(id);
	 uint16_t nStrips = (StripModuleGeom->specificTopology()).nstrips(); 
	 uint16_t nAPVs = nStrips/128;
	 
     RawDigis.clear();
     RawDigis.insert(RawDigis.begin(), nStrips, 0);
     
	 std::vector<uint16_t> stripsPerAPV;
	 stripsPerAPV.clear();
	 stripsPerAPV.insert(stripsPerAPV.begin(), nAPVs, 0);
	 edm::DetSet<SiStripDigi>::const_iterator itDigis = inDigis.begin();
     for(; itDigis != inDigis.end(); ++ itDigis){
    	 uint16_t strip = itDigis->strip(); 
         RawDigis[strip] = itDigis->adc();
         ++stripsPerAPV[strip/128];
      }
      
      for(uint16_t APV=0; APV<nAPVs; ++APV){
       		if(stripsPerAPV[APV]>64){
       			for(uint16_t strip = APV*128; strip < (APV+1)*128; ++strip) RawDigis[strip] = RawDigis[strip] * 2 - 1024;
       		}
       	}
}


//Suppressors Virgin Raw And Processed Raw
//--------------------------------------------------------
uint16_t SiStripRawProcessingAlgorithms::SuppressVirginRawData(const uint32_t& id, const uint16_t& firstAPV, std::vector<int16_t>& processedRawDigis , edm::DetSet<SiStripDigi>& suppressedDigis ){
      
      subtractorPed->subtract( id, firstAPV*128,processedRawDigis);
      return this->SuppressProcessedRawData(id, firstAPV, processedRawDigis , suppressedDigis );
 
}

uint16_t SiStripRawProcessingAlgorithms::SuppressVirginRawData(const edm::DetSet<SiStripRawDigi>& rawDigis, edm::DetSet<SiStripDigi>& suppressedDigis){
   
   std::vector<int16_t> RawDigis;
   RawDigis.clear();
   edm::DetSet<SiStripRawDigi>::const_iterator itrawDigis = rawDigis.begin();
   for(; itrawDigis != rawDigis.end(); ++itrawDigis) RawDigis.push_back(itrawDigis->adc());
   return this->SuppressVirginRawData(rawDigis.id, 0,RawDigis , suppressedDigis);
}
  



uint16_t SiStripRawProcessingAlgorithms::SuppressProcessedRawData(const uint32_t& id, const uint16_t& firstAPV, std::vector<int16_t>& processedRawDigis , edm::DetSet<SiStripDigi>& suppressedDigis ){
      std::vector<int16_t>  processedRawDigisPedSubtracted ;
      
      int16_t nAPVFlagged =0;
      if( doAPVRestore ) processedRawDigisPedSubtracted.assign(processedRawDigis.begin(), processedRawDigis.end());
      subtractorCMN->subtract(id, firstAPV,  processedRawDigis);
      if( doAPVRestore ) nAPVFlagged = restorer->InspectAndRestore(id, firstAPV, processedRawDigisPedSubtracted, processedRawDigis, subtractorCMN->getAPVsCM() );
      suppressor->suppress( processedRawDigis, firstAPV,  suppressedDigis ); 
      return nAPVFlagged;
}


uint16_t SiStripRawProcessingAlgorithms::SuppressProcessedRawData(const edm::DetSet<SiStripRawDigi>& rawDigis, edm::DetSet<SiStripDigi>& suppressedDigis){
   std::vector<int16_t> RawDigis;
   RawDigis.clear();
   edm::DetSet<SiStripRawDigi>::const_iterator itrawDigis = rawDigis.begin();
   for(; itrawDigis != rawDigis.end(); ++itrawDigis) RawDigis.push_back(itrawDigis->adc());
   return this->SuppressProcessedRawData(rawDigis.id, 0, RawDigis , suppressedDigis );
}


//Convert to Hybrid Format 
//--------------------------------------------------------

uint16_t SiStripRawProcessingAlgorithms::ConvertVirginRawToHybrid(const uint32_t& id, const uint16_t& firstAPV, std::vector<int16_t>& processedRawDigis , edm::DetSet<SiStripDigi>& suppressedDigis ){

      //std::vector<int16_t>  tmpVR;
      //tmpVR.clear();
      //tmpVR.assign(processedRawDigis.begin(), processedRawDigis.end());
      
      std::vector<int16_t>  processedRawDigisPedSubtracted;
      std::vector<bool> markedVRAPVs;
     
      for(uint16_t strip=0; strip < processedRawDigis.size(); ++strip) processedRawDigis[strip] += 1024;   //adding one MSB

      
      subtractorPed->subtract( id, firstAPV*128,processedRawDigis);      //all strips are pedestals subtracted
      
      for(uint16_t strip=0; strip < processedRawDigis.size(); ++strip) processedRawDigis[strip] /= 2;
      processedRawDigisPedSubtracted.assign(processedRawDigis.begin(), processedRawDigis.end());
      
      subtractorCMN->subtract(id, firstAPV,  processedRawDigis);      
      int16_t nAPVFlagged = restorer->InspectForHybridFormatEmulation(id, firstAPV, processedRawDigis, subtractorCMN->getAPVsCM(), markedVRAPVs);
      
      for(uint16_t strip=0; strip < processedRawDigis.size(); ++strip){
      	 processedRawDigis[strip] *= 2;
      }
      
      for(uint16_t APV=firstAPV ; APV< processedRawDigis.size()/128 + firstAPV; ++APV){
      	
      	if(markedVRAPVs[APV]){
      		//GB 23/6/08: truncation should be done at the very beginning
      		for(uint16_t strip = (APV-firstAPV)*128; strip < (APV-firstAPV+1)*128; ++strip) {
      			suppressedDigis.push_back(SiStripDigi(strip+firstAPV*128, ( processedRawDigisPedSubtracted[strip] < 0 ? 0 : suppressor->truncate(processedRawDigisPedSubtracted[strip]))));  
      		//	std::cout << "detID: " << id <<  " strip: " << strip <<  " ADC: " << tmpVR[strip] << std::endl;		   			
      		}
      	}else{
      		std::vector<int16_t> singleAPVdigi;
  		 	singleAPVdigi.clear();
  		 	for(int16_t strip = (APV-firstAPV)*128; strip < (APV-firstAPV+1)*128; ++strip) singleAPVdigi.push_back(processedRawDigis[strip]); 
      	  	suppressor->suppress(singleAPVdigi, APV,  suppressedDigis );
      	}
      }
       
      return nAPVFlagged;

}

uint16_t SiStripRawProcessingAlgorithms::ConvertVirginRawToHybrid(const edm::DetSet<SiStripRawDigi>& rawDigis, edm::DetSet<SiStripDigi>& suppressedDigis){
   
   std::vector<int16_t> RawDigis;
   RawDigis.clear();
   edm::DetSet<SiStripRawDigi>::const_iterator itrawDigis = rawDigis.begin();
   for(; itrawDigis != rawDigis.end(); ++itrawDigis) RawDigis.push_back(itrawDigis->adc());
   return this->ConvertVirginRawToHybrid(rawDigis.id, 0,RawDigis , suppressedDigis);
}
  



