/** \file
 * Implementation of class RawDataCollectorByLabel
 *
 */

#include "EventFilter/RawDataCollector/src/RawDataCollectorByLabel.h"
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

RawDataCollectorByLabel::RawDataCollectorByLabel(const edm::ParameterSet& pset) {

  inputTags_ = pset.getParameter<std::vector<InputTag> >("RawCollectionList");
  verbose_ = pset.getUntrackedParameter<int>("verbose",0);

  inputTokens_.reserve(inputTags_.size());
  for(tag_iterator_t inputTag = inputTags_.begin(); inputTag != inputTags_.end(); ++inputTag ) {
    inputTokens_.push_back(consumes<FEDRawDataCollection>(*inputTag));
  }
  produces<FEDRawDataCollection>();
}

RawDataCollectorByLabel::~RawDataCollectorByLabel(){

}


void RawDataCollectorByLabel::produce(Event & e, const EventSetup& c){

 /// Get Data from all FEDs
 std::vector< Handle<FEDRawDataCollection> > rawData;
 rawData.reserve(inputTokens_.size());
 for(tok_iterator_t inputTok = inputTokens_.begin(); inputTok != inputTokens_.end(); ++inputTok ) {
   Handle<FEDRawDataCollection> input;
   if (e.getByToken(*inputTok,input)){
     rawData.push_back(input);
   }
   //else{     //skipping the inputtag requested. but this is a normal operation to bare data & MC. silent warning   }
 }

 auto producedData = std::make_unique<FEDRawDataCollection>();

 for (unsigned int i=0; i< rawData.size(); ++i ) { 

   const FEDRawDataCollection *rdc=rawData[i].product();

   if ( verbose_ > 0 ) {
     std::cout << "\nRAW collection #" << i+1 << std::endl;
     std::cout << "branch name = " << rawData[i].provenance()->branchName() << std::endl;
     std::cout << "process index = " << rawData[i].provenance()->productID().processIndex() << std::endl;
   }

   for ( int j=0; j< FEDNumbering::MAXFEDID; ++j ) {
     const FEDRawData & fedData = rdc->FEDData(j);
     size_t size=fedData.size();

     if ( size > 0 ) {
       // this fed has data -- lets copy it
       if(verbose_ > 1) std::cout << "Copying data from FED #" << j << std::endl;
       FEDRawData & fedDataProd = producedData->FEDData(j);
       if ( fedDataProd.size() != 0 ) {
        if(verbose_ > 1) {
            std::cout << " More than one FEDRawDataCollection with data in FED ";
            std::cout << j << " Skipping the 2nd\n";
        }
        continue;
       } 
       fedDataProd.resize(size);
       unsigned char *dataProd=fedDataProd.data();
       const unsigned char *data=fedData.data();
       for ( unsigned int k=0; k<size; ++k ) {
         dataProd[k]=data[k];
       }
     }
   }
 }

 // Insert the new product in the event  
 e.put(std::move(producedData));

}


