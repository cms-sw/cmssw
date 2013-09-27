/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "RootDelayedReader.h"
#include "InputFile.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/WrapperOwningHolder.h"
#include "DataFormats/Common/interface/RefCoreStreamer.h"

#include "TROOT.h"
#include "TBranch.h"
#include "TClass.h"

#include <cassert>

namespace edm {

  RootDelayedReader::RootDelayedReader(
      RootTree const& tree,
      boost::shared_ptr<InputFile> filePtr,
      InputType inputType) :
   tree_(tree),
   filePtr_(filePtr),
   nextReader_(),
   inputType_(inputType) {
  }

  RootDelayedReader::~RootDelayedReader() {
  }

  WrapperOwningHolder
  RootDelayedReader::getProduct_(BranchKey const& k, WrapperInterfaceBase const* interface, EDProductGetter const* ep) const {
    iterator iter = branchIter(k);
    if (!found(iter)) {
      if (nextReader_) {
        return nextReader_->getProduct(k, interface, ep);
      } else {
        return WrapperOwningHolder();
      }
    }
    roottree::BranchInfo const& branchInfo = getBranchInfo(iter);
    TBranch* br = branchInfo.productBranch_;
    if (br == nullptr) {
      if (nextReader_) {
        return nextReader_->getProduct(k, interface, ep);
      } else {
        return WrapperOwningHolder();
      }
    }
   
    setRefCoreStreamer(ep);
    TClass* cp = branchInfo.classCache_;
    if(nullptr == cp) {
      branchInfo.classCache_ = gROOT->GetClass(branchInfo.branchDescription_.wrappedName().c_str());
      cp = branchInfo.classCache_;
    }
    void* p = cp->New();
    br->SetAddress(&p);
    tree_.getEntry(br, tree_.entryNumberForIndex(ep->transitionIndex()));
    if(tree_.branchType() == InEvent) {
      // CMS-THREADING For the primary input source calls to this function need to be serialized
      InputFile::reportReadBranch(inputType_, std::string(br->GetName()));
    }
    setRefCoreStreamer(false);
    WrapperOwningHolder edp(p, interface);
    return edp;
  }
}
