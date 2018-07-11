#ifndef FWCore_Framework_MergeableRunProductProcesses_h
#define FWCore_Framework_MergeableRunProductProcesses_h

#include <string>
#include <vector>

namespace edm {

  class ProductRegistry;

  class MergeableRunProductProcesses {
  public:

    MergeableRunProductProcesses();

    std::vector<std::string> const& processesWithMergeableRunProducts() const {
      return processesWithMergeableRunProducts_;
    }

    std::string const& getProcessName(unsigned int index) const {
      return processesWithMergeableRunProducts_[index];
    }

    std::vector<std::string>::size_type size() const {
      return processesWithMergeableRunProducts_.size();
    }

    // Called once in a job right after the ProductRegistry is frozen.
    // After that the names stored in this class are never modified.
    void setProcessesWithMergeableRunProducts(ProductRegistry const& productRegistry);

  private:

    // Holds the process names for processes that created mergeable
    // run products that are in the input of the current job.
    // Note this does not include the current process.
    std::vector<std::string> processesWithMergeableRunProducts_;
  };
}
#endif
