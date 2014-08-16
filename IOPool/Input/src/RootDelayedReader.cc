/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "RootDelayedReader.h"
#include "InputFile.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/RefCoreStreamer.h"

#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"
#include "FWCore/Framework/src/SharedResourcesRegistry.h"

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

  std::auto_ptr<EDProduct>
  RootDelayedReader::getProduct_(BranchKey const& k, EDProductGetter const* ep) const {
    iterator iter = branchIter(k);
    if (!found(iter)) {
      if (nextReader_) {
        return nextReader_->getProduct(k, ep);
      } else {
        return std::auto_ptr<EDProduct>();
      }
    }
    roottree::BranchInfo const& branchInfo = getBranchInfo(iter);
    TBranch* br = branchInfo.productBranch_;
    if (br == nullptr) {
      if (nextReader_) {
        return nextReader_->getProduct(k, ep);
      } else {
        return std::auto_ptr<EDProduct>();
      }
    }
   
    setRefCoreStreamer(ep);
    TClass* cp = branchInfo.classCache_;
    if(nullptr == cp) {
      branchInfo.classCache_ = gROOT->GetClass(branchInfo.branchDescription_.wrappedName().c_str());
      cp = branchInfo.classCache_;
      branchInfo.offsetToEDProduct_ = edProductClass_->GetBaseClassOffset(edProductClass_);
    }
    void* p = cp->New();

    // A union is used to avoid possible copies during the triple cast that would otherwise be needed. 	 
    // std::auto_ptr<EDProduct> edp(static_cast<EDProduct *>(static_cast<void *>(static_cast<unsigned char *>(p) + branchInfo.offsetToEDProduct_))); 	 
    union { 	 
      void* vp; 	 
      unsigned char* ucp; 	 
      EDProduct* edp; 	 
    } pointerUnion; 	 
    pointerUnion.vp = p; 	 
    pointerUnion.ucp += branchInfo.offsetToEDProduct_; 	 
    std::auto_ptr<EDProduct> edp(pointerUnion.edp);

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
