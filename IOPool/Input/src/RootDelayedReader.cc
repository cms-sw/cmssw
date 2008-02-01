/*----------------------------------------------------------------------
$Id: RootDelayedReader.cc,v 1.19 2008/01/30 00:28:29 wmtan Exp $
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
 : entryNumber_(entry), branches_(bMap), filePtr_(filePtr), fileFormatVersion_(fileFormatVersion) {}

  RootDelayedReader::~RootDelayedReader() {}

  std::auto_ptr<EDProduct>
  RootDelayedReader::getProduct(BranchKey const& k, EDProductGetter const* ep) const {
    SetRefStreamer(ep);
    input::EventBranchInfo const& branchInfo = getBranchInfo(k);
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
    TBranch *br = getProvenanceBranch(k);

    if (fileFormatVersion_.value_ <= 5) {
      std::auto_ptr<BranchEntryDescription> pb(new BranchEntryDescription); 
      BranchEntryDescription* ppb = pb.get();
      br->SetAddress(&ppb);
      br->GetEntry(entryNumber_);
      if (pb->status_ == BranchEntryDescription::CreatorNotRun) {
	return std::auto_ptr<EntryDescription>(0);
      }
      std::auto_ptr<EntryDescription> result = pb->convertToEntryDescription();
      EntryDescriptionRegistry::instance()->insertMapped(*result);
      br->SetAddress(0);
      return result;
    }

    EntryDescriptionID hash;
    EntryDescriptionID *phash = &hash;
    br->SetAddress(&phash);
    br->GetEntry(entryNumber_);
    if (hash == EntryDescription().id()) {
      return std::auto_ptr<EntryDescription>(0);
    }
    std::auto_ptr<EntryDescription> result(new EntryDescription);
    if (!EntryDescriptionRegistry::instance()->getMapped(hash, *result))
      throw edm::Exception(errors::EventCorruption)
	<< "Could not find EntryDescriptionID "
	<< hash
	<< " in the EntryDescriptionRegistry read from the input file";
    br->SetAddress(0);
    return result;
  }
}
