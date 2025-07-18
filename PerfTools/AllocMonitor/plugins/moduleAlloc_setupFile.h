#ifndef FWCore_Services_moduleAlloc_setupFile_h
#define FWCore_Services_moduleAlloc_setupFile_h

#include <string>
#include <atomic>
#include <vector>

namespace edm {
  class ActivityRegistry;
  namespace service::moduleAlloc {
    struct ThreadAllocInfo;

    class Filter {
    public:
      //a negative module id corresponds to an ES module
      Filter(std::vector<int> const* moduleIDs);
      //returns true if should keep this
      bool startOnThread(int moduleID) const;
      const ThreadAllocInfo* stopOnThread(int moduleID) const;

      bool startOnThread() const;
      const ThreadAllocInfo* stopOnThread() const;

      void setGlobalKeep(bool iShouldKeep);
      bool globalKeep() const { return globalKeep_.load(); }

      bool keepModuleInfo(int moduleID) const;

    private:
      std::atomic<bool> globalKeep_ = true;
      std::vector<int> const* moduleIDs_ = nullptr;
    };
    void setupFile(std::string const& iFileName, edm::ActivityRegistry&, Filter const*);
  }  // namespace service::moduleAlloc
}  // namespace edm

#endif
