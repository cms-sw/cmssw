#include <memory>
#include "boost/shared_ptr.hpp"
#include <fstream>

#include "CondFormats/CSCObjects/interface/CSCChamberTimeCorrections.h"
#include "CondFormats/DataRecord/interface/CSCChamberTimeCorrectionsRcd.h"
#include "OnlineDB/CSCCondDB/interface/CSCChamberTimeCorrectionsValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCCableRead.h"

CSCChamberTimeCorrectionsValues::CSCChamberTimeCorrectionsValues(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  bool isForMC = iConfig.getUntrackedParameter<bool>("isForMC",true);
  chamberObj = prefill(isForMC);
  setWhatProduced(this,&CSCChamberTimeCorrectionsValues::produceChamberTimeCorrections);
  findingRecord<CSCChamberTimeCorrectionsRcd>();
  //now do what ever other initialization is needed
}


CSCChamberTimeCorrectionsValues::~CSCChamberTimeCorrectionsValues()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete chamberObj;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCChamberTimeCorrectionsValues::ReturnType
CSCChamberTimeCorrectionsValues::produceChamberTimeCorrections(const CSCChamberTimeCorrectionsRcd& iRecord)
{
  //need a new object so to not be deleted at exit
  CSCChamberTimeCorrections* mydata=new CSCChamberTimeCorrections( *chamberObj );
  return mydata;
  
}

 void CSCChamberTimeCorrectionsValues::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }
