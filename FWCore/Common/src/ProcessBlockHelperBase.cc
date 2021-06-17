#include "FWCore/Common/interface/ProcessBlockHelperBase.h"

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/ProductLabels.h"
#include "FWCore/Utilities/interface/TypeID.h"

namespace edm {

  ProcessBlockHelperBase::~ProcessBlockHelperBase() = default;

  void ProcessBlockHelperBase::updateForNewProcess(ProductRegistry const& productRegistry,
                                                   std::string const& processName) {
    // Add the current process at the end if there are any
    // process blocks produced in the current process.
    for (auto const& product : productRegistry.productList()) {
      auto const& desc = product.second;
      if (desc.branchType() == InProcess && desc.produced() && !desc.transient()) {
        processesWithProcessBlockProducts_.emplace_back(processName);
        addedProcesses_.emplace_back(processName);
        return;
      }
    }
  }

  std::string ProcessBlockHelperBase::selectProcess(ProductRegistry const& productRegistry,
                                                    ProductLabels const& productLabels,
                                                    TypeID const& typeID) const {
    std::string processName(productLabels.process);
    std::string selectedProcess;

    unsigned int bestPosition = 0;
    for (auto const& prod : productRegistry.productList()) {
      BranchDescription const& desc = prod.second;
      if (desc.branchType() == InProcess && !desc.produced() && desc.present() &&
          desc.moduleLabel() == productLabels.module && desc.productInstanceName() == productLabels.productInstance &&
          desc.unwrappedTypeID() == typeID && (processName.empty() || processName == desc.processName())) {
        // This code is to select the latest matching process
        unsigned int position = 0;
        bool found = false;
        for (auto const& processFromHelper : processesWithProcessBlockProducts_) {
          if (processFromHelper == desc.processName()) {
            found = true;
            break;
          }
          ++position;
        }
        if (found && position >= bestPosition) {
          bestPosition = position;
          selectedProcess = desc.processName();
        }
      }
    }
    return selectedProcess;
  }

}  // namespace edm
