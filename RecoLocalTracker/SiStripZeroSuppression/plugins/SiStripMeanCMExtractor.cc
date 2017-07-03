// -*- C++ -*-
//
// Package:    SiStripMeanCMExtractor
// Class:      SiStripMeanCMExtractor
// 
/**\class SiStripMeanCMExtractor SiStripMeanCMExtractor.cc RecoLocalTracker/SiStripMeanCMExtractor/src/SiStripMeanCMExtractor.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Ivan Amos Cali,32 4-A08,+41227673039,
//         Created:  Wed Oct 13 11:50:47 CEST 2010
//
//


// system include files
#include <sstream>
#include <memory>
#include <list>
#include <algorithm>
#include <cassert>


// user include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDigi/interface/SiStripProcessedRawDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"


typedef std::map<uint32_t, std::vector<float> > CMMap;

class SiStripMeanCMExtractor : public edm::one::EDProducer<> {
   public:
      explicit SiStripMeanCMExtractor( const edm::ParameterSet&);
      ~SiStripMeanCMExtractor() override;

   private:
      void beginJob() override ;
      void produce(edm::Event&, const edm::EventSetup&) override;
      void endJob() override ;
      
	  void init(const edm::EventSetup&);
	  
	  edm::ESHandle<SiStripPedestals> pedestalHandle_;
      uint32_t pedestal_cache_id_;
  
	  void StoreMean(const edm::DetSetVector<SiStripProcessedRawDigi>& );
	  void ConvertMeanMapToDetSetVector(std::vector<edm::DetSet<SiStripProcessedRawDigi> >&);
	  void CMExtractorFromPedestals(const edm::DetSetVector<SiStripRawDigi>&,std::vector<edm::DetSet<SiStripProcessedRawDigi> >&);
	  edm::InputTag _inputTag;
	  std::string _Algorithm;
	  uint16_t _nEventsToUse;
	  uint16_t _actualEvent;
      
	  CMMap _CMMap; //it contains the sum of the CM calculated before. The normalization for the number of events it is done at the end when it is written in the DetSetVector.
};


SiStripMeanCMExtractor::SiStripMeanCMExtractor(const edm::ParameterSet& conf) : 
     _inputTag(conf.getParameter<edm::InputTag> ("CMCollection")),
	_Algorithm(conf.getParameter<std::string>("Algorithm")),
	_nEventsToUse(conf.getParameter<uint32_t>("NEvents"))
{

	 
    if(_nEventsToUse < 1) _nEventsToUse=1;
	produces< edm::DetSetVector<SiStripProcessedRawDigi> > ("MEANAPVCM"); 
}


SiStripMeanCMExtractor::~SiStripMeanCMExtractor()
{
 
}

void SiStripMeanCMExtractor::init(const edm::EventSetup& es){
     
	uint32_t p_cache_id = es.get<SiStripPedestalsRcd>().cacheIdentifier();
   	
	if(p_cache_id != pedestal_cache_id_) {
		es.get<SiStripPedestalsRcd>().get( pedestalHandle_ );
		pedestal_cache_id_ = p_cache_id;
	}
}

// ------------ method called to produce the data  ------------
void
SiStripMeanCMExtractor::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   
     
   //if(_actualEvent > _nEventsToUse) return;
   
   std::vector<edm::DetSet<SiStripProcessedRawDigi> > meancm;
   
   if(_Algorithm == "StoredCM"){
	edm::Handle< edm::DetSetVector<SiStripProcessedRawDigi> > inputCM;
	iEvent.getByLabel(_inputTag,inputCM);
	
	this->StoreMean(*inputCM);
	this->ConvertMeanMapToDetSetVector(meancm);
	
   } else if (_Algorithm == "Pedestals"){
     this->init(iSetup);
	 
     edm::Handle< edm::DetSetVector<SiStripRawDigi> > input;
     iEvent.getByLabel(_inputTag,input);
	 
     this->CMExtractorFromPedestals(*input,meancm);
   }
   
   ++_actualEvent;
   
    
	
	
    iEvent.put(std::make_unique<edm::DetSetVector<SiStripProcessedRawDigi>>(meancm),"MEANAPVCM");
   
}

void SiStripMeanCMExtractor::CMExtractorFromPedestals(const edm::DetSetVector<SiStripRawDigi>& input, std::vector<edm::DetSet<SiStripProcessedRawDigi> >& meancm){
	meancm.clear();
	meancm.reserve(15000);    
	
	 for ( edm::DetSetVector<SiStripRawDigi>::const_iterator 
	  rawDigis = input.begin(); rawDigis != input.end(); rawDigis++) {
         SiStripPedestals::Range detPedestalRange = pedestalHandle_->getRange(rawDigis->id);
		 edm::DetSet<SiStripProcessedRawDigi> MeanCMDetSet(rawDigis->id);
		
		for(uint16_t APV = 0; APV < rawDigis->size()/128; ++APV){
			uint16_t MinPed =0;
			for(uint16_t strip = APV*128; strip< (APV+1)*128; ++strip){
			  uint16_t ped =  (uint16_t)pedestalHandle_->getPed(strip,detPedestalRange);
			  if(ped < MinPed) MinPed = ped;
			}
			if(MinPed>128) MinPed=128;
			MeanCMDetSet.push_back(MinPed);
		}
		
		meancm.push_back(MeanCMDetSet);	
	}
}

void SiStripMeanCMExtractor::StoreMean(const edm::DetSetVector<SiStripProcessedRawDigi>& Input){
	
	uint32_t detId;
	CMMap::iterator itMap;
	edm::DetSetVector<SiStripProcessedRawDigi>::const_iterator itInput;
	
	for(itInput = Input.begin(); itInput != Input.end(); ++itInput){
		detId = itInput->id;
		itMap = _CMMap.find(detId);
		edm::DetSet<SiStripProcessedRawDigi>::const_iterator itCM;
		std::vector<float> MeanCMNValue;
		MeanCMNValue.clear();
		if(itMap!=_CMMap.end()){   //the detId was already found
		    std::vector< float >& MapContent = itMap->second;
			std::vector<float>::iterator itMapVector = MapContent.begin();
			for(itCM = itInput->begin(); itCM != itInput->end(); ++itCM, ++itMapVector){
				MeanCMNValue.push_back(itCM->adc() + *itMapVector); 
            }
			_CMMap.erase(itMap);
            _CMMap.insert(itMap, std::pair<uint32_t, std::vector<float> >(detId,MeanCMNValue));			
		} else {                 //no detId found
			for(itCM = itInput->begin(); itCM != itInput->end(); ++itCM) MeanCMNValue.push_back(itCM->adc()); 			
			_CMMap.insert(std::pair<uint32_t, std::vector<float> >(detId,MeanCMNValue));
		}
	}
  
}

void 
SiStripMeanCMExtractor::ConvertMeanMapToDetSetVector(std::vector<edm::DetSet<SiStripProcessedRawDigi> >& meancm){
	CMMap::iterator itMap;
	std::vector<float>::const_iterator itMapVector;
	
	meancm.clear();
	meancm.reserve(15000);    
	
    for(itMap = _CMMap.begin(); itMap != _CMMap.end(); ++itMap){
       edm::DetSet<SiStripProcessedRawDigi> MeanCMDetSet(itMap->first);
	   for(itMapVector = (itMap->second).begin(); itMapVector != (itMap->second).end(); ++itMapVector) MeanCMDetSet.push_back(*itMapVector/(float)_actualEvent);
	   meancm.push_back(MeanCMDetSet);	
	}	
    
}
void 
SiStripMeanCMExtractor::beginJob()
{
	_actualEvent =1;
		
	_CMMap.clear();
	
}

void 
SiStripMeanCMExtractor::endJob() {
    
	   
}

DEFINE_FWK_MODULE(SiStripMeanCMExtractor);

