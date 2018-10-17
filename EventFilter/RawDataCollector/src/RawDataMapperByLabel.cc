/** \file
 * Implementation of class RawDataMapperByLabel
 *
 */

#include "DataFormats/Provenance/interface/ProcessHistory.h" 
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h" 
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <iostream>

using namespace edm;

class RawDataMapperByLabel: public edm::stream::EDProducer<> {
public:
    
    ///Constructor
    RawDataMapperByLabel(const edm::ParameterSet& pset);
    
    ///Destructor
    ~RawDataMapperByLabel() override;
    
    void produce(edm::Event & e, const edm::EventSetup& c) override; 
    
    static void fillDescriptions(edm::ConfigurationDescriptions &);     
private:
    
    typedef std::vector<edm::InputTag>::const_iterator tag_iterator_t;
    typedef std::vector<edm::EDGetTokenT<FEDRawDataCollection> >::const_iterator tok_iterator_t;

    std::vector<edm::InputTag> inputTags_ ;
    std::vector<edm::EDGetTokenT<FEDRawDataCollection> > inputTokens_;
    
    edm::InputTag mainCollectionTag_ ;
    edm::InputTag filledCollectionName_;
    bool firstEvent_;
    

};


RawDataMapperByLabel::RawDataMapperByLabel(const edm::ParameterSet& pset)
 : inputTags_(pset.getParameter<std::vector<edm::InputTag>>("rawCollectionList")),
   mainCollectionTag_(pset.getParameter<edm::InputTag>("mainCollection")),
   filledCollectionName_(edm::InputTag("")),
   firstEvent_(true)
{

  inputTokens_.reserve(inputTags_.size());
  for(auto const& inputTag: inputTags_) {
    inputTokens_.push_back(consumes<FEDRawDataCollection>(inputTag));
  }

  produces<FEDRawDataCollection>();
   
}


RawDataMapperByLabel::~RawDataMapperByLabel(){

}


void RawDataMapperByLabel::produce(Event & e, const EventSetup& c){
 
 bool alreadyACollectionFilled= false;
 tag_iterator_t inputTag = inputTags_.begin();
 for(tok_iterator_t inputTok = inputTokens_.begin(); inputTok != inputTokens_.end(); ++inputTok, ++inputTag  ) {
   Handle<FEDRawDataCollection> input;
   if(e.getByToken(*inputTok,input)){
      if(input.isValid()){
         if(alreadyACollectionFilled) throw cms::Exception("BadInput") << "Two input collections are present." << std::endl
         << "Please make sure that the input dataset has only one FEDRawDataCollector collection filled";
         
         if(firstEvent_){  
            filledCollectionName_ = *inputTag; 
            alreadyACollectionFilled = true;
            firstEvent_= false;  
         }
            
         
         if(!(filledCollectionName_==*inputTag)) throw cms::Exception("BadInput") << "The filled collection has changed!";
            
         if(!(mainCollectionTag_==filledCollectionName_)) e.put(std::make_unique<FEDRawDataCollection>(*input.product()));
                      
      }
   }
 }
}

void RawDataMapperByLabel::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
    edm::ParameterSetDescription desc;
      
    desc.add<std::vector<edm::InputTag>>("rawCollectionList", {{"rawDataCollector"}, {"rawDataRepacker"}, {"rawDataReducedFormat"}});
    desc.add<edm::InputTag>("mainCollection", edm::InputTag("rawDataCollector"));
    
    descriptions.add("rawDataMapperByLabel", desc);
}

DEFINE_FWK_MODULE(RawDataMapperByLabel);

