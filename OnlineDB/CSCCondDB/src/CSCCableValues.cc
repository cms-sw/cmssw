#include <memory>
#include "boost/shared_ptr.hpp"
#include <fstream>

#include "CondFormats/CSCObjects/interface/CSCCables.h"
#include "CondFormats/DataRecord/interface/CSCCablesRcd.h"
#include "OnlineDB/CSCCondDB/interface/CSCCableValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCCableRead.h"

CSCCableValues::CSCCableValues(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  cableObj = fillCables();
  setWhatProduced(this,&CSCCableValues::produceCables);
  findingRecord<CSCCablesRcd>();
  //now do what ever other initialization is needed
}


CSCCableValues::~CSCCableValues()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete cableObj;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCCableValues::ReturnType
CSCCableValues::produceCables(const CSCCablesRcd& iRecord)
{
  //need a new object so to not be deleted at exit
  CSCCables* mydata=new CSCCables( *cableObj );
  return mydata;
  
}

 void CSCCableValues::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }
