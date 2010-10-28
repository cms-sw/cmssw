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
// $Id$
//
//


// system include files
#include <sstream>
#include <memory>
#include <list>
#include <algorithm>
#include <cassert>


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDigi/interface/SiStripProcessedRawDigi.h"


typedef std::map<uint32_t, std::vector<float> > CMMap;

class SiStripMeanCMExtractor : public edm::EDProducer {
   public:
      explicit SiStripMeanCMExtractor(const edm::ParameterSet&);
      ~SiStripMeanCMExtractor();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
	  void StoreMean(const edm::DetSetVector<SiStripProcessedRawDigi>& );
	  void ConvertMeanMapToDetSetVector(std::vector<edm::DetSet<SiStripProcessedRawDigi> >&);
	  
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


// ------------ method called to produce the data  ------------
void
SiStripMeanCMExtractor::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   
   if(_actualEvent > _nEventsToUse) return;
   
   if(_Algorithm == "StoredCM"){
	edm::Handle< edm::DetSetVector<SiStripProcessedRawDigi> > inputCM;
	iEvent.getByLabel(_inputTag,inputCM);
	
	this->StoreMean(*inputCM);
   }
   
   ++_actualEvent;
   
    std::vector<edm::DetSet<SiStripProcessedRawDigi> > meancm;
	this->ConvertMeanMapToDetSetVector(meancm);
	
	std::auto_ptr< edm::DetSetVector<SiStripProcessedRawDigi> > outputMeanCM(new edm::DetSetVector<SiStripProcessedRawDigi>(meancm) );
    iEvent.put( outputMeanCM,"MEANAPVCM");
   
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
	_actualEvent =0;
		
	_CMMap.clear();
	
}

void 
SiStripMeanCMExtractor::endJob() {
    
	   
}

DEFINE_FWK_MODULE(SiStripMeanCMExtractor);

