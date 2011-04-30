/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "RootDelayedReader.h"
#include "DataFormats/Common/interface/WrapperHolder.h"
#include "DataFormats/Common/interface/RefCoreStreamer.h"

#include "TROOT.h"
#include "TBranch.h"
#include "TClass.h"

namespace edm {

  RootDelayedReader::RootDelayedReader(EntryNumber const& entry,
      boost::shared_ptr<BranchMap const> bMap,
      RootTree const& tree,
      boost::shared_ptr<InputFile> filePtr,
      FileFormatVersion const& fileFormatVersion) :
   entryNumber_(entry),
   branches_(bMap),
   tree_(tree),
   filePtr_(filePtr),
   nextReader_(),
   fileFormatVersion_(fileFormatVersion) {}

  RootDelayedReader::~RootDelayedReader() {}

  WrapperHolder
  RootDelayedReader::getProduct_(BranchKey const& k, WrapperInterfaceBase const* interface, EDProductGetter const* ep) const {
    iterator iter = branchIter(k);
    if (!found(iter)) {
      if (nextReader_) {
        return nextReader_->getProduct(k, interface, ep);
      } else {
        return WrapperHolder();
      }
    }
    roottree::BranchInfo const& branchInfo = getBranchInfo(iter);
    TBranch* br = branchInfo.productBranch_;
    if (br == 0) {
      if (nextReader_) {
        return nextReader_->getProduct(k, interface, ep);
      } else {
        return WrapperHolder();
      }
    }
    setRefCoreStreamer(ep, !fileFormatVersion_.splitProductIDs(), !fileFormatVersion_.productIDIsInt());
    TClass* cp = branchInfo.classCache_;
    if(0 == cp) {
      branchInfo.classCache_ = gROOT->GetClass(branchInfo.branchDescription_.wrappedName().c_str());
      cp = branchInfo.classCache_;
    }
    void* p = cp->New();
    br->SetAddress(&p);
    tree_.getEntry(br, entryNumber_);
    setRefCoreStreamer(!fileFormatVersion_.splitProductIDs());
    WrapperHolder edp(p, interface, WrapperHolder::Owned);
    return edp;
  }
}
