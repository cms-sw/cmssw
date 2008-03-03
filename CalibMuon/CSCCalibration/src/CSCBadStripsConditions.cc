#include <memory>
#include "boost/shared_ptr.hpp"
#include <fstream>

#include "CondFormats/CSCObjects/interface/CSCBadStrips.h"
#include "CondFormats/DataRecord/interface/CSCBadStripsRcd.h"
#include "CalibMuon/CSCCalibration/interface/CSCBadStripsConditions.h"


CSCBadStripsConditions::CSCBadStripsConditions(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  cndbBadStrips = prefillBadStrips();
  setWhatProduced(this,&CSCBadStripsConditions::produceBadStrips);
  findingRecord<CSCBadStripsRcd>();
  //now do what ever other initialization is needed
}


CSCBadStripsConditions::~CSCBadStripsConditions()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete cndbBadStrips;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCBadStripsConditions::ReturnType
CSCBadStripsConditions::produceBadStrips(const CSCBadStripsRcd& iRecord)
{
  //need a new object so to not be deleted at exit
  CSCBadStrips* mydata=new CSCBadStrips( *cndbBadStrips );
  return mydata;
  
}

 void CSCBadStripsConditions::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }
