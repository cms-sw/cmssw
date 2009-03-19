/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "RootDelayedReader.h"
#include "DataFormats/Common/interface/RefCoreStreamer.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "TROOT.h"
#include "TClass.h"
#include "TBranch.h"

namespace edm {

  RootDelayedReader::RootDelayedReader(EntryNumber const& entry,
      boost::shared_ptr<BranchMap const> bMap,
      boost::shared_ptr<TFile const> filePtr,
      FileFormatVersion const& fileFormatVersion) :
   entryNumber_(entry),
   branches_(bMap),
   filePtr_(filePtr),
   nextReader_(),
   fileFormatVersion_(fileFormatVersion) {}

  RootDelayedReader::~RootDelayedReader() {}

  std::auto_ptr<EDProduct>
  RootDelayedReader::getProduct_(BranchKey const& k, EDProductGetter const* ep) const {
    iterator iter = branchIter(k);
    if (!found(iter)) {
      assert(nextReader_);
      return nextReader_->getProduct(k, ep);
    }
    input::BranchInfo const& branchInfo = getBranchInfo(iter);
    TBranch *br = branchInfo.productBranch_;
    if (br == 0) {
      assert(nextReader_);
      return nextReader_->getProduct(k, ep);
    }
    setRefCoreStreamer(ep, !fileFormatVersion_.splitProductIDs(), !fileFormatVersion_.productIDIsInt());
    TClass *cp = gROOT->GetClass(branchInfo.branchDescription_.wrappedName().c_str());
    std::auto_ptr<EDProduct> p(static_cast<EDProduct *>(cp->New()));
    EDProduct *pp = p.get();
    br->SetAddress(&pp);
    input::getEntry(br, entryNumber_);
    setRefCoreStreamer(!fileFormatVersion_.splitProductIDs());
    return p;
  }
}
