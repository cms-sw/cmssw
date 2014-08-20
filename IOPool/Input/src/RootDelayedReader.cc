/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "RootDelayedReader.h"
#include "InputFile.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/RefCoreStreamer.h"

#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"
#include "FWCore/Framework/src/SharedResourcesRegistry.h"

#include "IOPool/Common/interface/getEDProductPtr.h"

#include "TROOT.h"
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
   edProductClass_(gROOT->GetClass("edm::EDProduct")) {
  }

  RootDelayedReader::~RootDelayedReader() {
  }

  SharedResourcesAcquirer*
  RootDelayedReader::sharedResources_() const {
    return resourceAcquirer_.get();
  }

  std::unique_ptr<EDProduct>
  RootDelayedReader::getProduct_(BranchKey const& k, EDProductGetter const* ep) const {
    iterator iter = branchIter(k);
    if (!found(iter)) {
      if (nextReader_) {
        return nextReader_->getProduct(k, ep);
      } else {
        return std::unique_ptr<EDProduct>();
      }
    }
    roottree::BranchInfo const& branchInfo = getBranchInfo(iter);
    TBranch* br = branchInfo.productBranch_;
    if (br == nullptr) {
      if (nextReader_) {
        return nextReader_->getProduct(k, ep);
      } else {
        return std::unique_ptr<EDProduct>();
      }
    }
   
    setRefCoreStreamer(ep);
    TClass* cp = branchInfo.classCache_;
    if(nullptr == cp) {
      branchInfo.classCache_ = gROOT->GetClass(branchInfo.branchDescription_.wrappedName().c_str());
      cp = branchInfo.classCache_;
      branchInfo.offsetToEDProduct_ = cp->GetBaseClassOffset(edProductClass_);
    }
    void* p = cp->New();
    std::unique_ptr<EDProduct> edp = getEDProductPtr(p, branchInfo.offsetToEDProduct_); 
    br->SetAddress(&p);
    tree_.getEntry(br, tree_.entryNumberForIndex(ep->transitionIndex()));
    if(tree_.branchType() == InEvent) {
      // CMS-THREADING For the primary input source calls to this function need to be serialized
      InputFile::reportReadBranch(inputType_, std::string(br->GetName()));
    }
    setRefCoreStreamer(false);
    return edp;
  }
}
