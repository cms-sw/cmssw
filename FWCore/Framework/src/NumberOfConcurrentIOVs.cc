
#include "FWCore/Framework/src/NumberOfConcurrentIOVs.h"

#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/HCTypeTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <string>
#include <vector>

namespace edm {
  namespace eventsetup {

    NumberOfConcurrentIOVs::NumberOfConcurrentIOVs() : numberConcurrentIOVs_(1) {}

    void NumberOfConcurrentIOVs::initialize(ParameterSet const* optionsPset) {
      if (optionsPset) {  // this condition is false for SubProcesses
        numberConcurrentIOVs_ = optionsPset->getUntrackedParameter<unsigned int>("numberOfConcurrentIOVs");

        ParameterSet const& pset(optionsPset->getUntrackedParameterSet("forceNumberOfConcurrentIOVs"));
        std::vector<std::string> recordNames = pset.getParameterNames();
        for (auto const& recordName : recordNames) {
          EventSetupRecordKey recordKey(eventsetup::EventSetupRecordKey::TypeTag::findType(recordName));
          forceNumberOfConcurrentIOVs_[recordKey] = pset.getUntrackedParameter<unsigned int>(recordName);
        }
      }
    }

    void NumberOfConcurrentIOVs::initialize(EventSetupProvider const& eventSetupProvider) {
      eventSetupProvider.fillRecordsNotAllowingConcurrentIOVs(recordsNotAllowingConcurrentIOVs_);
    }

    unsigned int NumberOfConcurrentIOVs::numberOfConcurrentIOVs(EventSetupRecordKey const& eventSetupKey) const {
      auto iForce = forceNumberOfConcurrentIOVs_.find(eventSetupKey);
      if (iForce != forceNumberOfConcurrentIOVs_.end()) {
        return iForce->second;
      }
      if (recordsNotAllowingConcurrentIOVs_.find(eventSetupKey) != recordsNotAllowingConcurrentIOVs_.end()) {
        return 1;
      }
      return numberConcurrentIOVs_;
    }
  }  // namespace eventsetup
}  // namespace edm
