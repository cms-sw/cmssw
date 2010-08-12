/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "RootDelayedReader.h"
#include "DataFormats/Common/interface/RefCoreStreamer.h"
#include "TROOT.h"
#include "TClass.h"
#include "TBranch.h"

namespace edm {

  RootDelayedReader::RootDelayedReader(EntryNumber const& entry,
      boost::shared_ptr<BranchMap const> bMap,
      boost::shared_ptr<TTreeCache> treeCache,
      boost::shared_ptr<TFile> filePtr,
      FileFormatVersion const& fileFormatVersion) :
   entryNumber_(entry),
   branches_(bMap),
   treeCache_(treeCache),
   filePtr_(filePtr),
   nextReader_(),
   fileFormatVersion_(fileFormatVersion) {}

  RootDelayedReader::~RootDelayedReader() {}

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
    input::BranchInfo const& branchInfo = getBranchInfo(iter);
    TBranch *br = branchInfo.productBranch_;
    if (br == 0) {
      if (nextReader_) {
        return nextReader_->getProduct(k, ep);
      } else {
        return std::auto_ptr<EDProduct>();
      }
    }
    setRefCoreStreamer(ep, !fileFormatVersion_.splitProductIDs(), !fileFormatVersion_.productIDIsInt());
    TClass *cp = branchInfo.classCache_;
    if(0 == cp) {
      branchInfo.classCache_ = gROOT->GetClass(branchInfo.branchDescription_.wrappedName().c_str());
      cp = branchInfo.classCache_;
      TClass *edProductClass = gROOT->GetClass("edm::EDProduct");
      branchInfo.offsetToEDProduct_ = edProductClass->GetBaseClassOffset(edProductClass);
    }
    void *p = cp->New();

    // A union is used to avoid possible copies during the triple cast that would otherwise be needed.
    //std::auto_ptr<EDProduct> edp(static_cast<EDProduct *>(static_cast<void *>(static_cast<unsigned char *>(p) + branchInfo.offsetToEDProduct_)));
    union {
      void* vp;
      unsigned char* ucp;
      EDProduct* edp;
    } pointerUnion;
    pointerUnion.vp = p;
    pointerUnion.ucp += branchInfo.offsetToEDProduct_;
    std::auto_ptr<EDProduct> edp(pointerUnion.edp);

    br->SetAddress(&p);
    input::getEntryWithCache(br, entryNumber_, treeCache_.get(), filePtr_.get());
    setRefCoreStreamer(!fileFormatVersion_.splitProductIDs());
    return edp;
  }
}
