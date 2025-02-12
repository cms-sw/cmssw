#include "FWCore/Common/interface/ProcessBlockHelperBase.h"

#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/ProductLabels.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include <algorithm>
#include <iterator>

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

    long int bestPosition = 0;
    for (auto const& prod : productRegistry.productList()) {
      ProductDescription const& desc = prod.second;
      if (desc.branchType() == InProcess && !desc.produced() && desc.present() &&
          desc.moduleLabel() == productLabels.module && desc.productInstanceName() == productLabels.productInstance &&
          desc.unwrappedTypeID() == typeID && (processName.empty() || processName == desc.processName())) {
        // This code is to select the latest matching process
        auto found =
            std::find_if(processesWithProcessBlockProducts_.begin(),
                         processesWithProcessBlockProducts_.end(),
                         [&desc](auto const& processFromHelper) { return processFromHelper == desc.processName(); });
        if (found != processesWithProcessBlockProducts_.end()) {
          const auto position = std::distance(processesWithProcessBlockProducts_.begin(), found);
          static_assert(std::is_same_v<decltype(bestPosition),
                                       decltype(std::distance(processesWithProcessBlockProducts_.begin(), found))>);
          if (position >= bestPosition) {
            bestPosition = position;
            selectedProcess = desc.processName();
          }
        }
      }
    }
    return selectedProcess;
  }

}  // namespace edm
