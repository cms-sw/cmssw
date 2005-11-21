/** \file
 *
 *  $Date: 2005/11/10 18:55:03 $
 *  $Revision: 1.7.2.4 $
 *  \author S. Argiro - N. Amapane - M. Zanetti 
 */


#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <EventFilter/DTRawToDigi/src/DTUnpackingModule.h>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>

#include <CondFormats/DTMapping/interface/DTReadOutMapping.h>
#include <CondFormats/DataRecord/interface/DTReadOutMappingRcd.h>

#include <EventFilter/DTRawToDigi/src/DTDDUWords.h>
#include <EventFilter/DTRawToDigi/src/DTDDUUnpacker.h>
#include <EventFilter/DTRawToDigi/src/DTROS25Unpacker.h>
#include <EventFilter/DTRawToDigi/src/DTROS8Unpacker.h>


using namespace edm;
using namespace std;

#include <iostream>


#define SLINK_WORD_SIZE 8


DTUnpackingModule::DTUnpackingModule(const edm::ParameterSet& pset):
  dduUnpacker(new DTDDUUnpacker()),
  ros25Unpacker(new DTROS25Unpacker()),
  ros8Unpacker(new DTROS8Unpacker()) 
{
  produces<DTDigiCollection>();
}

DTUnpackingModule::~DTUnpackingModule(){
  delete dduUnpacker;
  delete ros25Unpacker;
  delete ros8Unpacker;
}


void DTUnpackingModule::produce(Event & e, const EventSetup& context){

  // Get the data from the event 
  Handle<FEDRawDataCollection> rawdata;
  e.getByLabel("DaqRawData", rawdata);

  // Get the mapping from the setup
  ESHandle<DTReadOutMapping> mapping;
  context.get<DTReadOutMappingRcd>().get(mapping);
  
  // Create the result i.e. the collection of MB Digis
  auto_ptr<DTDigiCollection> product(new DTDigiCollection);


  // Loop over the DT FEDs
  int dduID = 0;
  for (int id=FEDNumbering::getDTFEDIds().first; id<=FEDNumbering::getDTFEDIds().second; ++id){ 
    
    const FEDRawData& feddata = rawdata->FEDData(id);
    
    if (feddata.size()){
      
      // Unpack the DDU data
      dduUnpacker->interpretRawData(feddata.data(), feddata.size());

      // Unpack the ROS25 data
      ros25Unpacker->interpretRawData(feddata.data(), feddata.size(), dduID, mapping, product);

      // Unpack the ROS8 data
      ros8Unpacker->interpretRawData(feddata.data(), feddata.size(), mapping, product);

    }

    dduID++; 
  }

  // commit to the event  
  e.put(product);
}

