/* \file EcalDCCUnpackingModule.h
 *
 *  $Date: 2005/08/03 15:23:18 $
 *  $Revision: 1.1 $
 *  \author N. Marinelli 
 */

#include <EventFilter/EcalTBRawToDigi/interface/EcalDCCUnpackingModule.h>
#include <EventFilter/EcalTBRawToDigi/src/EcalTBDaqFormatter.h>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>

#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>

using namespace raw;
using namespace edm;
using namespace std;
using namespace cms;

#include <iostream>

EcalDCCUnpackingModule::EcalDCCUnpackingModule(const edm::ParameterSet& pset) : 
  formatter(new EcalTBDaqFormatter()) {}

EcalDCCUnpackingModule::~EcalDCCUnpackingModule(){
  delete formatter;
}


void EcalDCCUnpackingModule::produce(Event & e, const EventSetup& c){

  Handle<FEDRawDataCollection> rawdata;
  e.getByLabel("EcalDaqRawData", rawdata);
  
  // create the collection of Ecal Digis
  auto_ptr<EBDigiCollection> product(new EBDigiCollection);

  for (unsigned int id= 0; id<=FEDRawDataCollection::lastfedid; ++id){ 
    
    if ( id != 1 ) continue;
    cout << "EcalDCCUnpackingModule::Got FED ID "<<id <<" ";
    const FEDRawData& data = rawdata->FEDData(id);
    cout << " Fed data size " << data.data_.size() << endl;
    
    if (data.data_.size()){
      
      // do the conversion and fill the container
      formatter->interpretRawData(data,  *product );
    }// endif 
  }//endfor
  
  // commit to the event  
  e.put(product);
  

}
