#ifndef Framework_TestProcessor_ESProduceEntry_h
#define Framework_TestProcessor_ESProduceEntry_h
// -*- C++ -*-
//
// Package:     Framework/TestProcessor
// Class  :     ESProduceEntry
//
/**\class ESProduceEntry ESProduceEntry.h "ESProduceEntry.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  root
//         Created:  Tue, 08 May 2018 19:46:46 GMT
//

// system include files

// user include files

// forward declarations
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/DataKey.h"
#include "FWCore/Framework/interface/ESProductResolver.h"
#include <memory>
namespace edm {
  namespace test {
    struct ESProduceEntry {
      ESProduceEntry(edm::eventsetup::EventSetupRecordKey const& iRecKey,
                     edm::eventsetup::DataKey const& iDataKey,
                     std::shared_ptr<edm::eventsetup::ESProductResolver> iResolver)
          : recordKey_(iRecKey), dataKey_(iDataKey), resolver_(std::move(iResolver)) {}
      edm::eventsetup::EventSetupRecordKey recordKey_;
      edm::eventsetup::DataKey dataKey_;
      std::shared_ptr<edm::eventsetup::ESProductResolver> resolver_;
    };
  }  // namespace test
}  // namespace edm
#endif
