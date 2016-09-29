/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "RootDelayedReader.h"
#include "InputFile.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/RefCoreStreamer.h"

#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"
#include "FWCore/Framework/src/SharedResourcesRegistry.h"

#include "IOPool/Common/interface/getWrapperBasePtr.h"

#include "FWCore/Utilities/interface/EDMException.h"

#include "TBranch.h"
#include "TClass.h"

#include <cassert>

namespace edm {

  RootDelayedReader::RootDelayedReader(
      RootTree const& tree,
      std::shared_ptr<InputFile> filePtr,
      InputType inputType) :
   tree_(tree),
   filePtr_(filePtr),
   nextReader_(),
   resourceAcquirer_(inputType == InputType::Primary ? new SharedResourcesAcquirer() : static_cast<SharedResourcesAcquirer*>(nullptr)),
   inputType_(inputType),
   wrapperBaseTClass_(TClass::GetClass("edm::WrapperBase")) {
     if(inputType == InputType::Primary) {
       auto resources = SharedResourcesRegistry::instance()->createAcquirerForSourceDelayedReader();
       resourceAcquirer_=std::make_unique<SharedResourcesAcquirer>(std::move(resources.first));
       mutex_ = resources.second;
     }
  }

  RootDelayedReader::~RootDelayedReader() {
  }

  std::pair<SharedResourcesAcquirer*, std::recursive_mutex*>
  RootDelayedReader::sharedResources_() const {
    return std::make_pair(resourceAcquirer_.get(), mutex_.get());
  }

  std::unique_ptr<WrapperBase>
  RootDelayedReader::getProduct_(BranchKey const& k, EDProductGetter const* ep) {
    if (lastException_) {
      std::rethrow_exception(lastException_);
    }
    iterator iter = branchIter(k);
    if (!found(iter)) {
      if (nextReader_) {
        return nextReader_->getProduct(k, ep);
      } else {
        return std::unique_ptr<WrapperBase>();
      }
    }
    roottree::BranchInfo const& branchInfo = getBranchInfo(iter);
    TBranch* br = branchInfo.productBranch_;
    if (br == nullptr) {
      if (nextReader_) {
        return nextReader_->getProduct(k, ep);
      } else {
        return std::unique_ptr<WrapperBase>();
      }
    }
   
    setRefCoreStreamer(ep);
    //make code exception safe
    std::shared_ptr<void> refCoreStreamerGuard(nullptr,[](void*){    setRefCoreStreamer(false);
      ;});
    TClass* cp = branchInfo.classCache_;
    if(nullptr == cp) {
      branchInfo.classCache_ = TClass::GetClass(branchInfo.branchDescription_.wrappedName().c_str());
      cp = branchInfo.classCache_;
      branchInfo.offsetToWrapperBase_ = cp->GetBaseClassOffset(wrapperBaseTClass_);
    }
    void* p = cp->New();
    std::unique_ptr<WrapperBase> edp = getWrapperBasePtr(p, branchInfo.offsetToWrapperBase_); 
    br->SetAddress(&p);
    try{
      tree_.getEntry(br, tree_.entryNumberForIndex(ep->transitionIndex()));
    } catch(edm::Exception& exception) {
      exception.addContext("Rethrowing an exception that happened on a different thread.");
      lastException_ = std::current_exception();
    } catch(...) {
      lastException_ = std::current_exception();
    }
    if(lastException_) {
      std::rethrow_exception(lastException_);
    }
    if(tree_.branchType() == InEvent) {
      // CMS-THREADING For the primary input source calls to this function need to be serialized
      InputFile::reportReadBranch(inputType_, std::string(br->GetName()));
    }
    return edp;
  }
}
