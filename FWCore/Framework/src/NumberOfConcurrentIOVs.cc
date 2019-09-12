
#include "FWCore/Framework/src/NumberOfConcurrentIOVs.h"

#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/HCTypeTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <algorithm>
#include <cassert>
#include <string>

namespace edm {
  namespace eventsetup {

    NumberOfConcurrentIOVs::NumberOfConcurrentIOVs() : numberConcurrentIOVs_(1) {}

    void NumberOfConcurrentIOVs::readConfigurationParameters(ParameterSet const* eventSetupPset) {
      if (eventSetupPset) {  // this condition is false for SubProcesses
        numberConcurrentIOVs_ = eventSetupPset->getUntrackedParameter<unsigned int>("numberOfConcurrentIOVs");
        if (numberConcurrentIOVs_ == 0) {
          numberConcurrentIOVs_ = 1;
        }

        ParameterSet const& pset(eventSetupPset->getUntrackedParameterSet("forceNumberOfConcurrentIOVs"));
        std::vector<std::string> recordNames = pset.getParameterNames();
        forceNumberOfConcurrentIOVs_.reserve(recordNames.size());
        for (auto const& recordName : recordNames) {
          EventSetupRecordKey recordKey(eventsetup::EventSetupRecordKey::TypeTag::findType(recordName));
          forceNumberOfConcurrentIOVs_.emplace_back(recordKey, pset.getUntrackedParameter<unsigned int>(recordName));
        }
        std::sort(forceNumberOfConcurrentIOVs_.begin(),
                  forceNumberOfConcurrentIOVs_.end(),
                  [](auto const& left, auto const& right) { return left.first < right.first; });
      }
    }

    void NumberOfConcurrentIOVs::setMaxConcurrentIOVs(unsigned int nStreams, unsigned int nConcurrentLumis) {
      maxConcurrentIOVs_ = std::min(nStreams, nConcurrentLumis);
    }

    void NumberOfConcurrentIOVs::fillRecordsNotAllowingConcurrentIOVs(EventSetupProvider const& eventSetupProvider) {
      eventSetupProvider.fillRecordsNotAllowingConcurrentIOVs(recordsNotAllowingConcurrentIOVs_);
    }

    unsigned int NumberOfConcurrentIOVs::numberOfConcurrentIOVs(EventSetupRecordKey const& eventSetupKey,
                                                                bool printInfoMsg) const {
      assert(numberConcurrentIOVs_ != 0);
      auto iter = std::lower_bound(forceNumberOfConcurrentIOVs_.begin(),
                                   forceNumberOfConcurrentIOVs_.end(),
                                   std::make_pair(eventSetupKey, 0u),
                                   [](auto const& left, auto const& right) { return left.first < right.first; });
      if (iter != forceNumberOfConcurrentIOVs_.end() && iter->first == eventSetupKey) {
        if (printInfoMsg && iter->second > maxConcurrentIOVs_) {
          LogInfo("Configuration") << "For record " << eventSetupKey.name() << " you have configured " << iter->second
                                   << " concurrent IOVs.\n"
                                   << "But you cannot have more concurrent IOVs than lumis or streams.\n"
                                   << "There will not be more than " << maxConcurrentIOVs_ << " concurrent IOVs.\n";
        }
        return std::min(iter->second, maxConcurrentIOVs_);
      }
      if (recordsNotAllowingConcurrentIOVs_.find(eventSetupKey) != recordsNotAllowingConcurrentIOVs_.end()) {
        return 1;
      }
      if (printInfoMsg && numberConcurrentIOVs_ > maxConcurrentIOVs_) {
        LogInfo("Configuration") << "For record " << eventSetupKey.name() << " you have configured "
                                 << numberConcurrentIOVs_ << " concurrent IOVs.\n"
                                 << "But you cannot have more concurrent IOVs than lumis or streams.\n"
                                 << "There will not be more than " << maxConcurrentIOVs_ << " concurrent IOVs.\n";
      }
      return std::min(numberConcurrentIOVs_, maxConcurrentIOVs_);
    }

    void NumberOfConcurrentIOVs::clear() {
      // Mark this as invalid
      numberConcurrentIOVs_ = 0;

      // Free up memory
      recordsNotAllowingConcurrentIOVs_.clear();
      std::vector<std::pair<EventSetupRecordKey, unsigned int>>().swap(forceNumberOfConcurrentIOVs_);
    }
  }  // namespace eventsetup
}  // namespace edm
