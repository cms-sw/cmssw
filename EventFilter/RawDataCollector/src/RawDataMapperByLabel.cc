/** \file
 * Implementation of class RawDataMapperByLabel
 *
 */

#include "EventFilter/RawDataCollector/interface/RawDataMapperByLabel.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h" 
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h" 
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

using namespace edm;

RawDataMapperByLabel::RawDataMapperByLabel(const edm::ParameterSet& pset) {

  inputTags_ = pset.getParameter<std::vector<InputTag> >("RawCollectionList");
  mainCollectionTag_ = pset.getParameter<InputTag>("MainCollection");
  verbose_ = pset.getUntrackedParameter<int>("verbose",0);

  inputTokens_.reserve(inputTags_.size());
  for(tag_iterator_t inputTag = inputTags_.begin(); inputTag != inputTags_.end(); ++inputTag ) {
    inputTokens_.push_back(consumes<FEDRawDataCollection>(*inputTag));
  }
  produces<FEDRawDataCollection>();
}

RawDataMapperByLabel::~RawDataMapperByLabel(){

}


void RawDataMapperByLabel::produce(Event & e, const EventSetup& c){

 
 bool AlredyACollectionFilled= false;
 tag_iterator_t inputTag = inputTags_.begin();
 //unsigned int i=0;
 for(tok_iterator_t inputTok = inputTokens_.begin(); inputTok != inputTokens_.end(); ++inputTok, ++inputTag  ) {
   Handle<FEDRawDataCollection> input;
   if (e.getByToken(*inputTok,input)){
    	if(input.isValid()){
    		if(AlredyACollectionFilled) throw cms::Exception("Unknown input type") << "Two input collections are present. Please make sure that the input dataset has only one FEDRawDataCollector collection filled";
            //auto producedData = std::make_unique<FEDRawDataCollection>();
            //const FEDRawDataCollection *rdc=input.product();
            e.put(std::move(input.product()));
            AlredyACollectionFilled = true;            
   }
    
   }
   
  
   
 //  if(secondCollectionPresent) 
        
        
  
 }

 // Insert the new product in the event  

}


