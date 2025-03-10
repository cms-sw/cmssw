/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "RootDelayedReader.h"
#include "InputFile.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/RefCoreStreamer.h"

#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"
#include "FWCore/Framework/interface/SharedResourcesRegistry.h"

#include "IOPool/Common/interface/getWrapperBasePtr.h"

#include "FWCore/Utilities/interface/EDMException.h"

#include "TBranch.h"
#include "TClass.h"

#include <cassert>

namespace edm {

  RootDelayedReader::RootDelayedReader(RootTree const& tree, std::shared_ptr<InputFile> filePtr, InputType inputType)
      : tree_(tree),
        filePtr_(filePtr),
        nextReader_(),
        inputType_(inputType),
        wrapperBaseTClass_(TClass::GetClass("edm::WrapperBase")) {
    if (inputType == InputType::Primary) {
      auto resources = SharedResourcesRegistry::instance()->createAcquirerForSourceDelayedReader();
      resourceAcquirer_ = std::make_unique<SharedResourcesAcquirer>(std::move(resources.first));
      mutex_ = resources.second;
    }
  }

  RootDelayedReader::~RootDelayedReader() {}

  std::pair<SharedResourcesAcquirer*, std::recursive_mutex*> RootDelayedReader::sharedResources_() const {
    return std::make_pair(resourceAcquirer_.get(), mutex_.get());
  }

  std::shared_ptr<WrapperBase> RootDelayedReader::getProduct_(BranchID const& k, EDProductGetter const* ep) {
    if (lastException_) {
      try {
        std::rethrow_exception(lastException_);
      } catch (edm::Exception const& e) {
        //avoid growing the context each time the exception is rethrown.
        auto copy = e;
        copy.addContext("Rethrowing an exception that happened on a different read request.");
        throw copy;
      } catch (cms::Exception& e) {
        //If we do anything here to 'copy', we would lose the actual type of the exception.
        e.addContext("Rethrowing an exception that happened on a different read request.");
        throw;
      }
    }
    auto branchInfo = getBranchInfo(k);
    if (not branchInfo) {
      if (nextReader_) {
        return nextReader_->getProduct(k, ep);
      } else {
        return std::shared_ptr<WrapperBase>();
      }
    }
    TBranch* br = branchInfo->productBranch_;
    if (br == nullptr) {
      if (nextReader_) {
        return nextReader_->getProduct(k, ep);
      } else {
        return std::shared_ptr<WrapperBase>();
      }
    }

    setRefCoreStreamer(ep);
    //make code exception safe
    std::shared_ptr<void> refCoreStreamerGuard(nullptr, [](void*) {
      setRefCoreStreamer(false);
      ;
    });
    TClass* cp = branchInfo->classCache_;
    if (nullptr == cp) {
      branchInfo->classCache_ = TClass::GetClass(branchInfo->productDescription_.wrappedName().c_str());
      cp = branchInfo->classCache_;
      branchInfo->offsetToWrapperBase_ = cp->GetBaseClassOffset(wrapperBaseTClass_);
    }
    void* p = cp->New();
    std::unique_ptr<WrapperBase> edp = getWrapperBasePtr(p, branchInfo->offsetToWrapperBase_);
    br->SetAddress(&p);
    try {
      //Run, Lumi, and ProcessBlock only have 1 entry number, which is index 0
      tree_.getEntry(br, tree_.entryNumberForIndex(tree_.branchType() == InEvent ? ep->transitionIndex() : 0));
    } catch (...) {
      lastException_ = std::current_exception();
      std::rethrow_exception(lastException_);
    }
    if (tree_.branchType() == InEvent) {
      // CMS-THREADING For the primary input source calls to this function need to be serialized
      InputFile::reportReadBranch(inputType_, std::string(br->GetName()));
    }
    return edp;
  }
}  // namespace edm
