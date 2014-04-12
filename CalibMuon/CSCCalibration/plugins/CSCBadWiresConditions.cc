#include <memory>
#include "boost/shared_ptr.hpp"
#include <fstream>

#include "CondFormats/CSCObjects/interface/CSCBadWires.h"
#include "CondFormats/DataRecord/interface/CSCBadWiresRcd.h"
#include "CalibMuon/CSCCalibration/interface/CSCBadWiresConditions.h"


CSCBadWiresConditions::CSCBadWiresConditions(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  cndbBadWires = prefillBadWires();
  setWhatProduced(this,&CSCBadWiresConditions::produceBadWires);
  findingRecord<CSCBadWiresRcd>();
  //now do what ever other initialization is needed
}


CSCBadWiresConditions::~CSCBadWiresConditions()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete cndbBadWires;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCBadWiresConditions::ReturnType
CSCBadWiresConditions::produceBadWires(const CSCBadWiresRcd& iRecord)
{
  //need a new object so to not be deleted at exit
  CSCBadWires* mydata=new CSCBadWires( *cndbBadWires );
  return mydata;
  
}

 void CSCBadWiresConditions::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }
