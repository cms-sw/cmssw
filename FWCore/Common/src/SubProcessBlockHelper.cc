#include "FWCore/Common/interface/SubProcessBlockHelper.h"

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Utilities/interface/BranchType.h"

#include <cassert>
#include <string>

namespace edm {

  ProcessBlockHelperBase const* SubProcessBlockHelper::topProcessBlockHelper() const { return topProcessBlockHelper_; }

  std::vector<std::string> const& SubProcessBlockHelper::topProcessesWithProcessBlockProducts() const {
    return topProcessBlockHelper_->processesWithProcessBlockProducts();
  }

  unsigned int SubProcessBlockHelper::nProcessesInFirstFile() const {
    return topProcessBlockHelper_->nProcessesInFirstFile();
  }

  std::vector<std::vector<unsigned int>> const& SubProcessBlockHelper::processBlockCacheIndices() const {
    return topProcessBlockHelper_->processBlockCacheIndices();
  }

  std::vector<std::vector<unsigned int>> const& SubProcessBlockHelper::nEntries() const {
    return topProcessBlockHelper_->nEntries();
  }

  std::vector<unsigned int> const& SubProcessBlockHelper::cacheIndexVectorsPerFile() const {
    return topProcessBlockHelper_->cacheIndexVectorsPerFile();
  }

  std::vector<unsigned int> const& SubProcessBlockHelper::cacheEntriesPerFile() const {
    return topProcessBlockHelper_->cacheEntriesPerFile();
  }

  unsigned int SubProcessBlockHelper::processBlockIndex(
      std::string const& processName, EventToProcessBlockIndexes const& eventToProcessBlockIndexes) const {
    return topProcessBlockHelper_->processBlockIndex(processName, eventToProcessBlockIndexes);
  }

  unsigned int SubProcessBlockHelper::outerOffset() const { return topProcessBlockHelper_->outerOffset(); }

  void SubProcessBlockHelper::updateFromParentProcess(ProcessBlockHelperBase const& parentProcessBlockHelper,
                                                      ProductRegistry const& productRegistry) {
    topProcessBlockHelper_ = parentProcessBlockHelper.topProcessBlockHelper();

    // If a SubProcess keeps any ProcessBlock products from its parent process, then insert their
    // process names.
    assert(processesWithProcessBlockProducts().empty());
    for (auto const& processName : parentProcessBlockHelper.processesWithProcessBlockProducts()) {
      for (auto const& item : productRegistry.productList()) {
        BranchDescription const& desc = item.second;
        if (desc.branchType() == InProcess && desc.present() && desc.processName() == processName) {
          emplaceBackProcessName(processName);
          break;
        }
      }
    }

    // Repeat for addedProcesses
    assert(addedProcesses().empty());
    for (auto const& processName : parentProcessBlockHelper.addedProcesses()) {
      for (auto const& item : productRegistry.productList()) {
        BranchDescription const& desc = item.second;
        if (desc.branchType() == InProcess && desc.present() && desc.processName() == processName) {
          emplaceBackAddedProcessName(processName);
          break;
        }
      }
    }
  }

}  // namespace edm
