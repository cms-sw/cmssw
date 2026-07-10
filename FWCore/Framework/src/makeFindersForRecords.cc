#include "makeFindersForRecords.h"

#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/src/IntersectingIOVRecordIntervalFinder.h"
#include "FWCore/Framework/interface/DependentRecordIntervalFinder.h"
#include "FWCore/Framework/interface/SupportingRecordIntervalFinderHelper.h"
#include "FWCore/Framework/interface/RecordDependencyRegister.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "make_shared_noexcept_false.h"

#include <memory>
#include <vector>
#include <map>
namespace {
  using namespace edm;
  struct FinderConsolidator {
    explicit FinderConsolidator(edm::eventsetup::EventSetupRecordKey const& key)
        : key_(key), finders_(), finalFinder_() {}
    edm::eventsetup::EventSetupRecordKey key_;
    std::vector<edm::propagate_const<std::shared_ptr<EventSetupRecordIntervalFinder>>> finders_;
    std::shared_ptr<EventSetupRecordIntervalFinder> finalFinder_;
    std::set<eventsetup::EventSetupRecordKey> supportingRecords() const { return dependencies(key_); }

    std::shared_ptr<EventSetupRecordIntervalFinder> finalizeFinder() {
      if (1 < finders_.size()) {
        std::shared_ptr<eventsetup::IntersectingIOVRecordIntervalFinder> intFinder =
            make_shared_noexcept_false<eventsetup::IntersectingIOVRecordIntervalFinder>(key_);
        intFinder->swapFinders(finders_);
        finalFinder_ = intFinder;
      } else if (1 == finders_.size()) {
        finalFinder_ = edm::get_underlying<std::shared_ptr<EventSetupRecordIntervalFinder>>(finders_.front());
      }
      if (not supportingRecords().empty()) {
        //do this now as the return value of this routine is used to set the supporting finders of other records and we need to have the DependentRecordIntervalFinder in place before that is done
        std::shared_ptr<eventsetup::DependentRecordIntervalFinder> newFinder =
            make_shared_noexcept_false<eventsetup::DependentRecordIntervalFinder>(key_);
        newFinder->setAlternateFinder(finalFinder_);
        finalFinder_ = newFinder;
      }
      return finalFinder_;
    }

    ///If the provided Record depends on other Records, here are the supporting Finders
    void setSupportingFinders(const std::map<eventsetup::EventSetupRecordKey,
                                             std::shared_ptr<EventSetupRecordIntervalFinder>>& iKeyToFinders) {
      auto const& supportingKeys = supportingRecords();
      if (supportingKeys.empty()) {
        return;
      }
      assert(finalFinder_);
      auto depFinder = dynamic_cast<eventsetup::DependentRecordIntervalFinder*>(finalFinder_.get());
      assert(depFinder);

      std::string missingRecords;

      for (auto const& key : supportingKeys) {
        auto itFound = iKeyToFinders.find(key);
        if (itFound == iKeyToFinders.end()) {
          if (missingRecords.empty()) {
            missingRecords = key.name();
          } else {
            missingRecords += ", ";
            missingRecords += key.name();
          }
        } else {
          eventsetup::SupportingRecordIntervalFinderHelper helper(itFound->first, itFound->second);
          depFinder->addSupporter(helper);
        }
      }
      if (!missingRecords.empty()) {
        edm::LogInfo("EventSetupDependency")
            << "The EventSetup record " << key_.name() << " depends on at least one Record \n (" << missingRecords
            << ") which is not present in the job."
               "\n This may lead to an exception begin thrown during event processing.\n If no exception occurs "
               "during the job than it is usually safe to ignore this message.";
      }
    }
  };
}  // namespace
namespace edm::impl {
  std::map<edm::eventsetup::EventSetupRecordKey, std::shared_ptr<edm::EventSetupRecordIntervalFinder>>
  makeFindersForRecords(std::set<edm::eventsetup::EventSetupRecordKey> const& iKeys,
                        std::vector<std::shared_ptr<edm::EventSetupRecordIntervalFinder>> const& iFinders) {
    std::map<edm::eventsetup::EventSetupRecordKey, FinderConsolidator> consolidatorMap;
    //Need a consolidator for each key
    for (auto const& key : iKeys) {
      consolidatorMap.emplace(key, key);
    }
    //associate each finder with the keys it is finding for
    for (auto& finder : iFinders) {
      auto const& keys = finder->findingForRecords();
      for (auto const& key : keys) {
        consolidatorMap.at(key).finders_.push_back(finder);
      }
    }
    std::map<edm::eventsetup::EventSetupRecordKey, std::shared_ptr<edm::EventSetupRecordIntervalFinder>> returnValue;
    for (auto& [key, consolidator] : consolidatorMap) {
      returnValue[key] = consolidator.finalizeFinder();
    }
    //now that we have the final finders for each record, we can set the supporting finders for each record that has dependencies
    for (auto& [key, consolidator] : consolidatorMap) {
      consolidator.setSupportingFinders(returnValue);
    }

    return returnValue;
  }
}  // namespace edm::impl
