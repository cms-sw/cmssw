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
#include "FWCore/Framework/interface/ESProductResolverProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "FWCore/TestProcessor/interface/ESProduceEntry.h"

// forward declarations
namespace edm {
  namespace test {

    class EventSetupTestHelper : public eventsetup::ESProductResolverProvider, public EventSetupRecordIntervalFinder {
    public:
      EventSetupTestHelper(std::vector<ESProduceEntry>);
      EventSetupTestHelper(const EventSetupTestHelper&) = delete;
      const EventSetupTestHelper& operator=(const EventSetupTestHelper&) = delete;

      std::shared_ptr<eventsetup::ESProductResolver> getResolver(unsigned int index);

      void resetAllProxies();

    protected:
      void setIntervalFor(const eventsetup::EventSetupRecordKey&, const IOVSyncValue&, ValidityInterval&) final;

      KeyedResolversVector registerProxies(const eventsetup::EventSetupRecordKey&, unsigned int iovIndex) final;

    private:
      // ---------- member data --------------------------------
      std::vector<ESProduceEntry> proxies_;
    };
  }  // namespace test
}  // namespace edm
#endif
