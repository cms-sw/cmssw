// -*- C++ -*-
//
// Package:     Framework
// Class  :     NoRecordException
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Aug 28 13:27:25 CDT 2009
//

// system include files

// user include files
#include "FWCore/Framework/interface/NoRecordException.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/EventSetupImpl.h"
#include "FWCore/Utilities/interface/Exception.h"

bool
edm::eventsetup::recordDoesExist( EventSetupImpl const& iES, EventSetupRecordKey const& iKey) {
  return iES.recordIsProvidedByAModule(iKey);
}


void
edm::eventsetup::no_record_exception_message_builder(cms::Exception& oException, const char* iName, bool iKnownRecord) {
   oException
   << "No \"" 
   << iName
   << "\" record found in the EventSetup.n";
  if(iKnownRecord) {
   oException<<"\n The Record is delivered by an ESSource or ESProducer but there is no valid IOV for the synchronizatio value.\n"
    " Please check \n"
    "   a) if the synchronization value is reasonable and report to the hypernews if it is not.\n"
    "   b) else check that all ESSources have been properly configured. \n";
  } else {
   oException <<"\n Please add an ESSource or ESProducer that delivers such a record.\n";
  }
}
