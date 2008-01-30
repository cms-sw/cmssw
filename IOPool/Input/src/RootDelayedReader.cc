/*----------------------------------------------------------------------
$Id: RootDelayedReader.cc,v 1.18 2007/12/31 10:18:17 elmer Exp $
----------------------------------------------------------------------*/

#include "RootDelayedReader.h"
#include "IOPool/Common/interface/RefStreamer.h"
#include "DataFormats/Provenance/interface/BranchEntryDescription.h"
#include "DataFormats/Provenance/interface/BranchKey.h"
#include "DataFormats/Provenance/interface/EntryDescription.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "TROOT.h"
#include "TClass.h"
#include "TBranch.h"

namespace edm {

  RootDelayedReader::RootDelayedReader(EntryNumber const& entry,
 boost::shared_ptr<BranchMap const> bMap,
 boost::shared_ptr<TFile const> filePtr,
 FileFormatVersion const& fileFormatVersion)
 : entryNumber_(entry), branches_(bMap), filePtr_(filePtr), fileFormatVersion_(fileFormatVersion) {}

  RootDelayedReader::~RootDelayedReader() {}

  std::auto_ptr<EDProduct>
  RootDelayedReader::getProduct(BranchKey const& k, EDProductGetter const* ep) const {
    SetRefStreamer(ep);
    input::EventBranchInfo const& branchInfo = branches().find(k)->second;
    TBranch *br = branchInfo.productBranch_;
    TClass *cp = gROOT->GetClass(branchInfo.branchDescription_.wrappedName().c_str());
    std::auto_ptr<EDProduct> p(static_cast<EDProduct *>(cp->New()));
    EDProduct *pp = p.get();
    br->SetAddress(&pp);
    br->GetEntry(entryNumber_);
    return p;
  }

  std::auto_ptr<EntryDescription>
  RootDelayedReader::getProvenance(BranchKey const& k) const {
    TBranch *br = branches().find(k)->second.provenanceBranch_;
    if (fileFormatVersion_.value_ <= 5) {
      std::auto_ptr<BranchEntryDescription> pb(new BranchEntryDescription); 
      BranchEntryDescription *ppb = pb.get();
      br->SetAddress(&ppb);
      br->GetEntry(entryNumber_);
      return pb->convertToEntryDescription(); 
    }
    std::auto_ptr<EntryDescription> p(new EntryDescription); 
    EntryDescription *pp = p.get();
    br->SetAddress(&pp);
    br->GetEntry(entryNumber_);
    return p;
  }
}
