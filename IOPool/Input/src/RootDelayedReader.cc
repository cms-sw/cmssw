/*----------------------------------------------------------------------
$Id: RootDelayedReader.cc,v 1.23 2008/04/18 01:57:58 wmtan Exp $
----------------------------------------------------------------------*/

#include "RootDelayedReader.h"
#include "IOPool/Common/interface/RefStreamer.h"
#include "DataFormats/Provenance/interface/BranchEntryDescription.h"
#include "DataFormats/Provenance/interface/EntryDescription.h"
#include "DataFormats/Provenance/interface/EntryDescriptionID.h"
#include "DataFormats/Provenance/interface/EntryDescriptionRegistry.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "TROOT.h"
#include "TClass.h"
#include "TBranch.h"

namespace edm {

  RootDelayedReader::RootDelayedReader(EntryNumber const& entry,
 boost::shared_ptr<BranchMap const> bMap,
 boost::shared_ptr<TFile const> filePtr,
 FileFormatVersion const& fileFormatVersion)
 : entryNumber_(entry), branches_(bMap), filePtr_(filePtr), fileFormatVersion_(fileFormatVersion), nextReader_() {}

  RootDelayedReader::~RootDelayedReader() {}

  std::auto_ptr<EDProduct>
  RootDelayedReader::getProduct_(BranchKey const& k, EDProductGetter const* ep) const {
    SetRefStreamer(ep);
    iterator iter = branchIter(k);
    if (!found(iter)) {
      assert(nextReader_);
      return nextReader_->getProduct(k, ep);
    }
    input::EventBranchInfo const& branchInfo = getBranchInfo(iter);
    TBranch *br = branchInfo.productBranch_;
    if (br == 0) {
      assert(nextReader_);
      return nextReader_->getProduct(k, ep);
    }
    TClass *cp = gROOT->GetClass(branchInfo.branchDescription_.wrappedName().c_str());
    std::auto_ptr<EDProduct> p(static_cast<EDProduct *>(cp->New()));
    EDProduct *pp = p.get();
    br->SetAddress(&pp);
    br->GetEntry(entryNumber_);
    return p;
  }

  std::auto_ptr<EntryDescription>
  RootDelayedReader::getProvenance_(BranchKey const& k) const {
    iterator iter = branchIter(k);
    if (!found(iter)) {
      assert(nextReader_);
      return nextReader_->getProvenance(k);
    }
    TBranch *br = getProvenanceBranch(iter);

    if (fileFormatVersion_.value_ <= 5) {
      std::auto_ptr<BranchEntryDescription> pb(new BranchEntryDescription); 
      BranchEntryDescription* ppb = pb.get();
      br->SetAddress(&ppb);
      br->GetEntry(entryNumber_);
      std::auto_ptr<EntryDescription> result = pb->convertToEntryDescription();
      EntryDescriptionRegistry::instance()->insertMapped(*result);
      br->SetAddress(0);
      return result;
    }

    EntryDescriptionID hash;
    EntryDescriptionID *phash = &hash;
    br->SetAddress(&phash);
    br->GetEntry(entryNumber_);
    std::auto_ptr<EntryDescription> result(new EntryDescription);
    if (!EntryDescriptionRegistry::instance()->getMapped(hash, *result))
      //  throw edm::Exception(errors::EventCorruption)
      // 	<< "Could not find EntryDescriptionID "
      // 	<< hash
      // 	<< " in the EntryDescriptionRegistry read from the input file";
    br->SetAddress(0);
    return result;
  }
}
