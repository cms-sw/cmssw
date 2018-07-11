
#include "FWCore/Framework/interface/MergeableRunProductProcesses.h"

#include "DataFormats/Common/interface/WrapperBase.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Utilities/interface/getAnyPtr.h"

#include "TClass.h"

#include <memory>
#include <set>

namespace edm {

  MergeableRunProductProcesses::MergeableRunProductProcesses() { }

  void MergeableRunProductProcesses::setProcessesWithMergeableRunProducts(ProductRegistry const& productRegistry) {
    TClass* wrapperBaseTClass = TypeWithDict::byName("edm::WrapperBase").getClass();
    std::set<std::string> processSet;
    for (auto const& prod : productRegistry.productList()) {
      BranchDescription const& desc = prod.second;
      if (desc.branchType() == InRun && !desc.produced() && desc.present()) {
        TClass* cp = desc.wrappedType().getClass();
        void* p = cp->New();
        int offset = cp->GetBaseClassOffset(wrapperBaseTClass);
        std::unique_ptr<WrapperBase> edp = getAnyPtr<WrapperBase>(p, offset);
        if (edp->isMergeable()) {
          processSet.insert(desc.processName());
        }
      }
    }
    processesWithMergeableRunProducts_.assign(processSet.begin(), processSet.end());
  }
}
