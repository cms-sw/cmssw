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
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <EventFilter/DTRawToDigi/plugins/DTUnpackingModule.h>
#include <DataFormats/DTDigi/interface/DTControlData.h>

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


DTUnpackingModule::DTUnpackingModule(const edm::ParameterSet& ps) : unpacker(0),dataType("") {

  dataType = ps.getParameter<string>("dataType");

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
  performDataIntegrityMonitor = unpackerParameters.getUntrackedParameter<bool>("performDataIntegrityMonitor",false); // default: false
  
  if(!dqmOnly) {
    produces<DTDigiCollection>();
    produces<DTLocalTriggerCollection>();
  }
  if(performDataIntegrityMonitor) {
    produces<std::vector<DTDDUData> >();
    produces<std::vector<std::vector<DTROS25Data> > >();
  }
}

DTUnpackingModule::~DTUnpackingModule(){
  delete unpacker;
}

void DTUnpackingModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("dataType","DDU");
  desc.add<edm::InputTag>("inputLabel",edm::InputTag("rawDataCollector"));
  desc.add<bool>("useStandardFEDid",true);
  desc.addUntracked<int>("minFEDid",770);
  desc.addUntracked<int>("maxFEDid",779);
  desc.addOptional<bool>("fedbyType");  // never used, only kept here for back-compatibility
  {
    edm::ParameterSetDescription psd0;
    psd0.addUntracked<bool>("debug",false);
    {
      edm::ParameterSetDescription psd1;
      psd1.addUntracked<bool>("writeSC",true);
      psd1.addUntracked<bool>("readingDDU",true);
      psd1.addUntracked<bool>("performDataIntegrityMonitor",false);
      psd1.addUntracked<bool>("readDDUIDfromDDU",true);
      psd1.addUntracked<bool>("debug",false);
      psd1.addUntracked<bool>("localDAQ",false);
      psd0.add<edm::ParameterSetDescription>("rosParameters",psd1);
    }
    psd0.addUntracked<bool>("performDataIntegrityMonitor",false);
    psd0.addUntracked<bool>("localDAQ",false);
    desc.add<edm::ParameterSetDescription>("readOutParameters",psd0);
  }
  desc.add<bool>("dqmOnly",false);
  descriptions.add("dtUnpackingModule",desc);
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

  auto_ptr<std::vector<DTDDUData> > dduProduct(new std::vector<DTDDUData>);
  auto_ptr<DTROS25Collection> ros25Product(new DTROS25Collection);
  
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
      if(performDataIntegrityMonitor) {
        if(dataType == "DDU") {
          dduProduct->push_back(dynamic_cast<DTDDUUnpacker*>(unpacker)->getDDUControlData());
          ros25Product->push_back(dynamic_cast<DTDDUUnpacker*>(unpacker)->getROSsControlData());
        }
        else if(dataType == "ROS25") {
          ros25Product->push_back(dynamic_cast<DTROS25Unpacker*>(unpacker)->getROSsControlData());
        }
      }
    }
  }

  // commit to the event  
  if(!dqmOnly) {
    e.put(detectorProduct);
    e.put(triggerProduct);
  }
  if(performDataIntegrityMonitor) {
    e.put(dduProduct);
    e.put(ros25Product);
  }
}

