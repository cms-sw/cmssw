
#include "DataFormats/Common/interface/setIsMergeable.h"

#include "DataFormats/Common/interface/WrapperBase.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/getAnyPtr.h"

#include "TClass.h"

#include <memory>

namespace edm {

  void setIsMergeable(BranchDescription& desc) {
    // Save some time here with the knowledge that the isMergeable
    // data member can only be true for run or lumi products.
    // It defaults to false. Also if it is true that means it
    // was already set.
    // Set it only for branches that are present
    if (desc.present() and (desc.branchType() == InRun or desc.branchType() == InLumi)) {
      if (!desc.isMergeable()) {
        TClass* wrapperBaseTClass = TypeWithDict::byName("edm::WrapperBase").getClass();
        TClass* tClass = desc.wrappedType().getClass();
        void* p = tClass->New();
        int offset = tClass->GetBaseClassOffset(wrapperBaseTClass);
        std::unique_ptr<WrapperBase> wrapperBase = getAnyPtr<WrapperBase>(p, offset);
        if (wrapperBase->isMergeable()) {
          desc.setIsMergeable(true);
        }
      }
    }
  }
}  // namespace edm
