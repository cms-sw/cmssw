#ifndef FWCore_TestProcessor_EventSetupTestHelper_h
#define FWCore_TestProcessor_EventSetupTestHelper_h
// -*- C++ -*-
//
// Package:     FWCore/TestProcessor
// Class  :     EventSetupTestHelper
// 
/**\class EventSetupTestHelper EventSetupTestHelper.h "EventSetupTestHelper.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  root
//         Created:  Tue, 08 May 2018 18:33:09 GMT
//

// system include files
#include <vector>

// user include files
#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "FWCore/TestProcessor/interface/ESProduceEntry.h"

// forward declarations
namespace edm {
namespace test {
  
  class EventSetupTestHelper : public eventsetup::DataProxyProvider, public EventSetupRecordIntervalFinder
{

   public:
  EventSetupTestHelper(std::vector<ESProduceEntry>);

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
  void newInterval(const eventsetup::EventSetupRecordKey& iRecordType,
                   const ValidityInterval& iInterval) final;

  std::shared_ptr<eventsetup::DataProxy> getProxy(unsigned int index );
  
  void resetAllProxies();
protected:
  void setIntervalFor(const eventsetup::EventSetupRecordKey&,
                      const IOVSyncValue& ,
                      ValidityInterval&) final;

  void registerProxies(const eventsetup::EventSetupRecordKey& iRecordKey ,
                       KeyedProxies& aProxyList) final ;


   private:
      EventSetupTestHelper(const EventSetupTestHelper&) = delete; // stop default

      const EventSetupTestHelper& operator=(const EventSetupTestHelper&) = delete; // stop default

      // ---------- member data --------------------------------
  std::vector<ESProduceEntry> proxies_;
};
}
}


#endif
