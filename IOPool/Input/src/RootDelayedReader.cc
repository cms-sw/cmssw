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
   resourceAcquirer_(inputType == InputType::Primary ? new SharedResourcesAcquirer(SharedResourcesRegistry::instance()->createAcquirerForSourceDelayedReader()) : static_cast<SharedResourcesAcquirer*>(nullptr)),
   inputType_(inputType),
   wrapperBaseTClass_(TClass::GetClass("edm::WrapperBase")) {
  }

  RootDelayedReader::~RootDelayedReader() {
  }

  SharedResourcesAcquirer*
  RootDelayedReader::sharedResources_() const {
    return resourceAcquirer_.get();
  }

  std::unique_ptr<WrapperBase>
  RootDelayedReader::getProduct_(BranchKey const& k, EDProductGetter const* ep) const {
    if (lastException_) {
      throw *lastException_;
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
    } catch(const edm::Exception& exception) {
      lastException_ = std::make_unique<Exception>(exception);
      lastException_->addContext("Rethrowing an exception that happened on a different thread.");
      throw exception;
    }
    if(tree_.branchType() == InEvent) {
      // CMS-THREADING For the primary input source calls to this function need to be serialized
      InputFile::reportReadBranch(inputType_, std::string(br->GetName()));
    }
    return edp;
  }
}
