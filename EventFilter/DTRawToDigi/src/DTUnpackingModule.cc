/** \file
 *
 *  $Date: 2005/11/22 13:52:15 $
 *  $Revision: 1.9 $
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

#include <CondFormats/DTObjects/interface/DTReadOutMapping.h>
#include <CondFormats/DataRecord/interface/DTReadOutMappingRcd.h>

#include <EventFilter/DTRawToDigi/src/DTDDUWords.h>
#include <EventFilter/DTRawToDigi/src/DTDDUUnpacker.h>
//#include <EventFilter/DTRawToDigi/src/DTROS25Unpacker.h>
#include <EventFilter/DTRawToDigi/src/DTROS8Unpacker.h>


using namespace edm;
using namespace std;

#include <iostream>


#define SLINK_WORD_SIZE 8


DTUnpackingModule::DTUnpackingModule(const edm::ParameterSet& pset) :
  unpacker(0)
{
  const string &  dataType = pset.getParameter<string>("dataType");

  if (dataType == "DDU") {
    unpacker = new DTDDUUnpacker();
  } else if (dataType == "ROS8") {
    unpacker = new DTROS8Unpacker();
  } else {
    throw cms::Exception("InvalidParameter") << "DTUnpackingModule: dataType "
					     << dataType << " is unknown";
  }

  produces<DTDigiCollection>();
}

DTUnpackingModule::~DTUnpackingModule(){
  delete unpacker;
}


void DTUnpackingModule::produce(Event & e, const EventSetup& context){

  // Get the data from the event 
  Handle<FEDRawDataCollection> rawdata;
  e.getByLabel("DaqRawData", rawdata);

  // Get the mapping from the setup
  ESHandle<DTReadOutMapping> mapping;
  //  context.get<DTReadOutMappingRcd>().get(mapping);
  
  // Create the result i.e. the collection of MB Digis
  auto_ptr<DTDigiCollection> product(new DTDigiCollection);


  // Loop over the DT FEDs
  for (int id=FEDNumbering::getDTFEDIds().first; id<=FEDNumbering::getDTFEDIds().second; ++id){ 
    
    const FEDRawData& feddata = rawdata->FEDData(id);
    
    if (feddata.size()){
      
      // Unpack the DDU data
      unpacker->interpretRawData(feddata.data(), feddata.size(), id, mapping, product);
    }
  }

  // commit to the event  
  e.put(product);
}

