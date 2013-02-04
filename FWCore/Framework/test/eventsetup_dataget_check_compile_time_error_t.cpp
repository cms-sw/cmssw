/*
 *  eventsetup_dataget_check_compile_time_error_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 4/7/05.
 *
 */

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/ESHandle.h"

using namespace edm;
class DataWithNoDefaultRecord {};

int main() {
   eventsetup::EventSetupProvider provider;
   EventSetup const& eventSetup = provider.eventSetupForInstance(IOVSyncValue(nullptr));
   //This should cause a compile time failure since DataWithNoDefaultRecord
   /// does not have a default record assigned
   ESHandle<DataWithNoDefaultRecord> pData;
   eventSetup.getData(pData);
   
   return 0;
}
