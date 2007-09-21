/*----------------------------------------------------------------------
$Id: RootDelayedReader.cc,v 1.12 2007/06/14 22:02:14 wmtan Exp $
----------------------------------------------------------------------*/

#include "RootDelayedReader.h"
#include "IOPool/Common/interface/RefStreamer.h"
#include "DataFormats/Provenance/interface/BranchKey.h"
#include "DataFormats/Provenance/interface/BranchEntryDescription.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "Reflex/Type.h"
#include "Reflex/Object.h"
#include "TBranch.h"

namespace edm {

  RootDelayedReader::RootDelayedReader(EntryNumber const& entry,
 boost::shared_ptr<BranchMap const> bMap,
 boost::shared_ptr<TFile const> filePtr)
 : entryNumber_(entry), branches_(bMap), filePtr_(filePtr) {}

  RootDelayedReader::~RootDelayedReader() {}

  std::auto_ptr<EDProduct>
  RootDelayedReader::getProduct(BranchKey const& k, EDProductGetter const* ep) const {
    SetRefStreamer(ep);
    input::EventBranchInfo const& branchInfo = branches().find(k)->second;
    TBranch *br = branchInfo.productBranch_;
    ROOT::Reflex::Object object = branchInfo.branchDescription_.type().Construct();
    std::auto_ptr<EDProduct> p(static_cast<EDProduct *>(object.Address()));
    EDProduct *pp = p.get();
    br->SetAddress(&pp);
    br->GetEntry(entryNumber_);
    return p;
  }

  std::auto_ptr<BranchEntryDescription>
  RootDelayedReader::getProvenance(BranchKey const& k) const {
    TBranch *br = branches().find(k)->second.provenanceBranch_;
    std::auto_ptr<BranchEntryDescription> p(new BranchEntryDescription); 
    BranchEntryDescription *pp = p.get();
    br->SetAddress(&pp);
    br->GetEntry(entryNumber_);
    return p;
  }
}
