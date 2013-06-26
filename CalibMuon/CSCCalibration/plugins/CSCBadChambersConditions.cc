#include <memory>
#include "boost/shared_ptr.hpp"
#include <fstream>

#include "CondFormats/CSCObjects/interface/CSCBadChambers.h"
#include "CondFormats/DataRecord/interface/CSCBadChambersRcd.h"
#include "CalibMuon/CSCCalibration/interface/CSCBadChambersConditions.h"


CSCBadChambersConditions::CSCBadChambersConditions(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  cndbBadChambers = prefillBadChambers();
  setWhatProduced(this,&CSCBadChambersConditions::produceBadChambers);
  findingRecord<CSCBadChambersRcd>();
  //now do what ever other initialization is needed
}


CSCBadChambersConditions::~CSCBadChambersConditions()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete cndbBadChambers;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCBadChambersConditions::ReturnType
CSCBadChambersConditions::produceBadChambers(const CSCBadChambersRcd& iRecord)
{
  //need a new object so to not be deleted at exit
  CSCBadChambers* mydata=new CSCBadChambers( *cndbBadChambers );
  return mydata;
  
}

 void CSCBadChambersConditions::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }
