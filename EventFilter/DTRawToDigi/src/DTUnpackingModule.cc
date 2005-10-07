/** \file
 *
 *  $Date: 2005/08/23 09:30:32 $
 *  $Revision: 1.3 $
 *  \author S. Argiro - N. Amapane 
 */

#include <EventFilter/DTRawToDigi/src/DTUnpackingModule.h>
#include <EventFilter/DTRawToDigi/src/DTDaqCMSFormatter.h>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>
#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>

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
  e.getByLabel("DaqRawData", rawdata);

  // create the collection of MB Digis
  auto_ptr<DTDigiCollection> product(new DTDigiCollection);

  
  for (int id=FEDNumbering::getDTFEDIds().first; id<=FEDNumbering::getDTFEDIds().second; ++id){ 

    const FEDRawData& data = rawdata->FEDData(id);

    if (data.size()){
      // do the conversion and fill the container
      formatter->interpretRawData(data, *product);
    }
  }
  
  // commit to the event  
  e.put(product);
  
}

