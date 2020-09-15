#include "FWCore/Common/interface/ProcessBlockHelper.h"

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/StoredProcessBlockHelper.h"
#include "FWCore/Utilities/interface/BranchType.h"

#include <cassert>

namespace edm {

  void ProcessBlockHelper::initializeFromPrimaryInput(ProductRegistry const& productRegistry,
                                                      StoredProcessBlockHelper const& storedProcessBlockHelper) {
    // do nothing if this is not the first input file
    if (!initializedFromInput_) {
      // copy in the process names from the input file in the same order
      // except remove the processes where all process block products were
      // dropped
      assert(processesWithProcessBlockProducts_.empty());
      for (auto const& processName : storedProcessBlockHelper.processesWithProcessBlockProducts()) {
        for (auto const& item : productRegistry.productList()) {
          BranchDescription const& desc = item.second;
          if (desc.branchType() == InProcess && desc.present() && desc.processName() == processName) {
            processesWithProcessBlockProducts_.emplace_back(processName);
            break;
          }
        }
      }
      initializedFromInput_ = true;
    }
  }

  void ProcessBlockHelper::updateForNewProcess(ProductRegistry const& productRegistry, std::string const& processName) {
    // Add the current process at the end if there are any
    // process blocks produced in the current process.
    for (auto const& product : productRegistry.productList()) {
      auto const& desc = product.second;
      if (desc.branchType() == InProcess && desc.produced() && !desc.transient()) {
        processesWithProcessBlockProducts_.emplace_back(processName);
        return;
      }
    }
  }

  void ProcessBlockHelper::updateFromParentProcess(ProcessBlockHelper const& parentProcessBlockHelper,
                                                   ProductRegistry const& productRegistry) {
    // This function is used by SubProcesses.
    // If a SubProcess keeps any ProcessBlock products from its parent process, then insert their
    // process names.
    assert(processesWithProcessBlockProducts_.empty());
    for (auto const& processName : parentProcessBlockHelper.processesWithProcessBlockProducts_) {
      for (auto const& item : productRegistry.productList()) {
        BranchDescription const& desc = item.second;
        if (desc.branchType() == InProcess && desc.present() && desc.processName() == processName) {
          processesWithProcessBlockProducts_.emplace_back(processName);
          break;
        }
      }
    }
  }

  void ProcessBlockHelper::updateAfterProductSelection(
      std::set<std::string> const& processesWithKeptProcessBlockProducts,
      ProcessBlockHelper const& processBlockHelper) {
    // Copy the list of processes with ProcessBlock products from the EventProcessor or SubProcess,
    // except remove any processes where the output module calling this has dropped all of those
    // products. We want to maintain the same order and only remove elements.
    assert(processesWithProcessBlockProducts_.empty());
    for (auto const& processCandidate : processBlockHelper.processesWithProcessBlockProducts()) {
      if (processesWithKeptProcessBlockProducts.find(processCandidate) != processesWithKeptProcessBlockProducts.end()) {
        processesWithProcessBlockProducts_.emplace_back(processCandidate);
      }
    }
  }

}  // namespace edm
