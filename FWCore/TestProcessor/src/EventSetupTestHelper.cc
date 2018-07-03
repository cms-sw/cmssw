// -*- C++ -*-
//
// Package:     FWCore/TestProcessor
// Class  :     EventSetupTestHelper
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  root
//         Created:  Tue, 08 May 2018 18:33:15 GMT
//

// system include files

// user include files
#include "FWCore/TestProcessor/interface/EventSetupTestHelper.h"
#include "FWCore/Framework/interface/DataProxy.h"

namespace edm {
  namespace test {
    
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
EventSetupTestHelper::EventSetupTestHelper(std::vector<ESProduceEntry> iProxies):
  proxies_{std::move(iProxies)}
{
  //Deal with duplicates
  std::set<eventsetup::EventSetupRecordKey> records;
  for(auto const& p: proxies_) {
    records.insert(p.recordKey_);
  }
  for(auto const&k: records) {
    usingRecordWithKey(k);
    findingRecordWithKey(k);
  }
}
    
void
EventSetupTestHelper::newInterval(const eventsetup::EventSetupRecordKey& iRecordType,
                                  const ValidityInterval& )  {
}

void
EventSetupTestHelper::setIntervalFor(const eventsetup::EventSetupRecordKey&,
                                     const IOVSyncValue& iSync,
                                     ValidityInterval& oIOV) {
  if(iSync.luminosityBlockNumber()==0) {
    //make valid through first run
    oIOV = ValidityInterval(iSync,
                            IOVSyncValue(EventID(iSync.eventID().run(),1,1)));
  } else if(iSync.eventID().event() == 0) {
    oIOV = ValidityInterval(iSync,
                            IOVSyncValue(EventID(iSync.eventID().run(),iSync.eventID().luminosityBlock(),1)));
  } else {
    //Make valid for only this point
    oIOV = ValidityInterval(iSync,iSync);
  }
}
    
void
EventSetupTestHelper::registerProxies(const eventsetup::EventSetupRecordKey& iRecordKey ,
                                      KeyedProxies& aProxyList) {
  for(auto const& p: proxies_) {
    if(p.recordKey_ == iRecordKey) {
      aProxyList.emplace_back(p.dataKey_,p.proxy_);
    }
  }
}

std::shared_ptr<eventsetup::DataProxy>
EventSetupTestHelper::getProxy(unsigned int iIndex) {
  return proxies_[iIndex].proxy_;
}
    
void
EventSetupTestHelper::resetAllProxies() {
  for(auto const& p: proxies_) {
    p.proxy_->invalidate();
  }
}


  }
}
