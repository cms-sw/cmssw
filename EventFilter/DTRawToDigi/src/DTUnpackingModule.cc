/* \file DTUnpackingModule.h
 *
 *  $Date: 2005/07/14 08:00:50 $
 *  $Revision: 1.2 $
 *  \author S. Argiro - N. Amapane 
 */

#include <EventFilter/DTRawToDigi/interface/DTUnpackingModule.h>
#include <EventFilter/DTRawToDigi/src/DTDaqCMSFormatter.h>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <DataFormats/DTDigis/interface/DTDigiCollection.h>
#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>

using namespace raw;
using namespace edm;
using namespace std;

#include <iostream>

DTUnpackingModule::DTUnpackingModule(const edm::ParameterSet& pset) : 
  formatter(new DTDaqCMSFormatter()) {
  produces<DTDigiCollection>();
}

DTUnpackingModule::~DTUnpackingModule(){
  delete formatter;
}


void DTUnpackingModule::produce(Event & e, const EventSetup& c){

  Handle<FEDRawDataCollection> rawdata;
  cout << " getting product " << endl;
  e.getByLabel("DaqRawData", rawdata);
  cout << " done " << endl;

  // create the collection of MB Digis
  auto_ptr<DTDigiCollection> product(new DTDigiCollection);

  for (unsigned int id= 0; id<=FEDRawDataCollection::lastfedid; ++id){ 

    // std::cout << "DTUnpackingModule::Got FED ID "<<id <<" ";
//     std::cout << "   data size: " << rawdata->getFedData(id).size() 
// 	      << std::endl;
    const FEDRawData& data = rawdata->FEDData(id);

    if (data.data_.size()){
      
//       // Dirty hack: recreate a DaqFEDRawData to feed the formatter
//       DaqFEDRawData d(const_cast<char*>(&(*data.begin())), data.size());

      // do the conversion and fill the container
      formatter->interpretRawData(data, *product);
    }// endif 
  }//endfor
  
  // commit to the event  
  e.put(product);
  
}

