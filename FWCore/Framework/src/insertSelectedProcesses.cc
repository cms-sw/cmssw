#include "FWCore/Framework/interface/insertSelectedProcesses.h"

#include "DataFormats/Common/interface/WrapperBase.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/getAnyPtr.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

#include "TClass.h"

#include <memory>

namespace edm {

  void insertSelectedProcesses(BranchDescription const& desc,
                               std::set<std::string>& processes) {

    // Select input processes in which mergeable run products were produced
    if (desc.branchType() == InRun && !desc.produced()) {

      // Determine if the product is "mergeable"
      TClass* tClass = desc.wrappedType().getClass();
      void* p = tClass->New();
      TClass* wrapperBaseTClass = TypeWithDict::byName("edm::WrapperBase").getClass();
      int offset = tClass->GetBaseClassOffset(wrapperBaseTClass);
      std::unique_ptr<WrapperBase> edp = getAnyPtr<WrapperBase>(p, offset);
      if (edp->isMergeable()) {
        // record the process names in a set (which is ordered and unique)
        processes.insert(desc.processName());
      }
    }
  }
}
