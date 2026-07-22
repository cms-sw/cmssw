#ifndef FWCore_Framework_makeFindersForRecords_h
#define FWCore_Framework_makeFindersForRecords_h
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include <memory>
#include <vector>
#include <map>
namespace edm::impl {
  std::map<edm::eventsetup::EventSetupRecordKey, std::shared_ptr<edm::EventSetupRecordIntervalFinder>>
  makeFindersForRecords(std::set<edm::eventsetup::EventSetupRecordKey> const& iKeys,
                        std::vector<std::shared_ptr<edm::EventSetupRecordIntervalFinder>> const& iFinders);
}  // namespace edm::impl
#endif
