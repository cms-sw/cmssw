#ifndef FWCore_Framework_SupportingRecordIntervalFinderHelper_h
#define FWCore_Framework_SupportingRecordIntervalFinderHelper_h
/** \class edm::eventsetup::SupportingRecordIntervalFinderHelper
*  Helper class for DependentRecordIntervalFinder to find intervals for supporting records.
*/

#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"

#include <cassert>
#include <memory>

namespace edm {
  namespace eventsetup {
    class SupportingRecordIntervalFinderHelper {
    public:
      SupportingRecordIntervalFinderHelper() = delete;
      SupportingRecordIntervalFinderHelper(EventSetupRecordKey const& iKey,
                                           std::shared_ptr<EventSetupRecordIntervalFinder> iFinder)
          : key_{iKey}, finder_{iFinder} {
        assert(finder_);
      }
      SupportingRecordIntervalFinderHelper(SupportingRecordIntervalFinderHelper const&) = default;
      SupportingRecordIntervalFinderHelper& operator=(SupportingRecordIntervalFinderHelper const&) = default;
      SupportingRecordIntervalFinderHelper(SupportingRecordIntervalFinderHelper&&) = default;
      SupportingRecordIntervalFinderHelper& operator=(SupportingRecordIntervalFinderHelper&&) = default;
      ~SupportingRecordIntervalFinderHelper() = default;

      const ValidityInterval& findIntervalFor(edm::IOVSyncValue const& iSync) {
        return finder_->findIntervalFor(key_, iSync);
      }

      bool operator<(SupportingRecordIntervalFinderHelper const& iRHS) const { return key_ < iRHS.key_; }

    private:
      EventSetupRecordKey key_;
      std::shared_ptr<EventSetupRecordIntervalFinder> finder_;
    };
  }  // namespace eventsetup
}  // namespace edm

#endif  // FWCore_Framework_SupportingRecordIntervalFinderHelper_h
