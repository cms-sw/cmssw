/** \file
 *
 *  \author S. Argiro - N. Amapane - M. Zanetti 
 * FRC 060906
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <FWCore/Framework/interface/Event.h>
#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <EventFilter/DTRawToDigi/plugins/DTUnpackingModule.h>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>
#include <DataFormats/DTDigi/interface/DTLocalTriggerCollection.h>

#include <CondFormats/DataRecord/interface/DTReadOutMappingRcd.h>

#include <EventFilter/DTRawToDigi/plugins/DTDDUUnpacker.h>
#include <EventFilter/DTRawToDigi/plugins/DTROS25Unpacker.h>
#include <EventFilter/DTRawToDigi/plugins/DTROS8Unpacker.h>


using namespace edm;
using namespace std;



#define SLINK_WORD_SIZE 8 


DTUnpackingModule::DTUnpackingModule(const edm::ParameterSet& ps) : unpacker(0) {

  const string & dataType = ps.getParameter<string>("dataType");

  ParameterSet unpackerParameters = ps.getParameter<ParameterSet>("readOutParameters");
  

  if (dataType == "DDU") {
    unpacker = new DTDDUUnpacker(unpackerParameters);
  } 
  else if (dataType == "ROS25") {
    unpacker = new DTROS25Unpacker(unpackerParameters.getParameter<ParameterSet>("rosParameters"));
  } 
  else if (dataType == "ROS8") {
    unpacker = new DTROS8Unpacker(unpackerParameters);
  } 
  else {
    throw cms::Exception("InvalidParameter") << "DTUnpackingModule: dataType "
					     << dataType << " is unknown";
  }

  inputLabel = consumes<FEDRawDataCollection>(ps.getParameter<InputTag>("inputLabel")); // default was: source
  useStandardFEDid_ = ps.getParameter<bool>("useStandardFEDid"); // default was: true
  minFEDid_ = ps.getUntrackedParameter<int>("minFEDid",770); // default: 770
  maxFEDid_ = ps.getUntrackedParameter<int>("maxFEDid",779); // default 779
  dqmOnly = ps.getParameter<bool>("dqmOnly"); // default: false

  if(!dqmOnly) {
    produces<DTDigiCollection>();
    produces<DTLocalTriggerCollection>();
  }
}

DTUnpackingModule::~DTUnpackingModule(){
  delete unpacker;
}


void DTUnpackingModule::produce(Event & e, const EventSetup& context){

  Handle<FEDRawDataCollection> rawdata;
  e.getByToken(inputLabel, rawdata);

  if(!rawdata.isValid()){
    LogError("DTUnpackingModule::produce") << " unable to get raw data from the event" << endl;
    return;
  }

  // Get the mapping from the setup
  ESHandle<DTReadOutMapping> mapping;
  context.get<DTReadOutMappingRcd>().get(mapping);
  
  // Create the result i.e. the collections of MB Digis and SC local triggers
  auto_ptr<DTDigiCollection> detectorProduct(new DTDigiCollection);
  auto_ptr<DTLocalTriggerCollection> triggerProduct(new DTLocalTriggerCollection);


  // Loop over the DT FEDs
  int FEDIDmin = 0, FEDIDMax = 0;
  if (useStandardFEDid_){
    FEDIDmin = FEDNumbering::MINDTFEDID;
    FEDIDMax = FEDNumbering::MAXDTFEDID;
  }
  else {
    FEDIDmin = minFEDid_;
    FEDIDMax = maxFEDid_;
  }
  
  for (int id=FEDIDmin; id<=FEDIDMax; ++id){ 
    
    const FEDRawData& feddata = rawdata->FEDData(id);
    
    if (feddata.size()){
      
      // Unpack the data
      unpacker->interpretRawData(reinterpret_cast<const unsigned int*>(feddata.data()), 
 				 feddata.size(), id, mapping, detectorProduct, triggerProduct);
    }
  }

  // commit to the event  
  if(!dqmOnly) {
    e.put(detectorProduct);
    e.put(triggerProduct);
  }
}

