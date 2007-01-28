#include "IOPool/Input/src/RootTree.h"
#include "IOPool/Input/src/RootDelayedReader.h"
#include "DataFormats/Common/interface/Provenance.h"
#include "DataFormats/Common/interface/BranchEntryDescription.h"
#include "FWCore/Framework/interface/DataBlockImpl.h"

#include <iostream>

namespace edm {
  RootTree::RootTree(boost::shared_ptr<TFile> filePtr, BranchType const& branchType) :
    filePtr_(filePtr),
    tree_(dynamic_cast<TTree *>(filePtr_.get() != 0 ? filePtr->Get(BranchTypeToProductTreeName(branchType).c_str()) : 0)),
    metaTree_(dynamic_cast<TTree *>(filePtr_.get() != 0 ? filePtr->Get(BranchTypeToMetaDataTreeName(branchType).c_str()) : 0)),

    auxBranch_(tree_ ? tree_->GetBranch(BranchTypeToAuxiliaryBranchName(branchType).c_str()): 0),
    entries_(tree_ ? tree_->GetEntries() : 0),
    entryNumber_(-1),
    origEntryNumber_(),
    branchNames_(),
    provenance_(),
    provenancePtrs_(),
    branches_(new BranchMap),
    products_()
  {
    int nBranches = (metaTree_ != 0 ? metaTree_->GetNbranches() : 0);
    if (nBranches > 0) {
      provenance_.reserve(nBranches);
      provenancePtrs_.reserve(nBranches);
    }
  }

  bool
  RootTree::isValid() const {
    if (metaTree_ == 0) {
      return tree_ != 0 && auxBranch_ != 0 &&
	 tree_->GetNbranches() == 1; 
    }
    return tree_ != 0 && auxBranch_ != 0 &&
	entries_ == metaTree_->GetEntries() &&
	 tree_->GetNbranches() <= metaTree_->GetNbranches() + 1; 
  }

  void
  RootTree::addBranch(BranchKey const& key,
		      BranchDescription const& prod,
		      std::string const& oldBranchName) {
      prod.init();
      //use the translated branch name 
      prod.provenancePresent_ = (metaTree_->GetBranch(oldBranchName.c_str()) != 0);
      TBranch * branch = tree_->GetBranch(oldBranchName.c_str());
      prod.present_ = (branch != 0);
      if (prod.provenancePresent()) {
        std::string const &name = prod.className();
        std::string const className = wrappedClassName(name);
        if (branch != 0) branches_->insert(std::make_pair(key, std::make_pair(className, branch)));
        products_.insert(std::make_pair(prod.productID(), prod));
	//we want the new branch name for the JobReport
	branchNames_.push_back(prod.branchName());
        int n = provenance_.size();
        provenance_.push_back(BranchEntryDescription());
        provenancePtrs_.push_back(&provenance_[n]);
        metaTree_->SetBranchAddress(oldBranchName.c_str(),(&provenancePtrs_[n]));
      }
  }

  void
  RootTree::fillGroups(DataBlockImpl& item) {
    if (metaTree_ == 0) return;
    // Loop over provenance
    metaTree_->GetEntry(entryNumber_);
    std::vector<BranchEntryDescription>::const_iterator pit = provenance_.begin();
    std::vector<BranchEntryDescription>::const_iterator pitEnd = provenance_.end();
    for (; pit != pitEnd; ++pit) {
      // if (pit->creatorStatus() != BranchEntryDescription::Success) continue;
      // BEGIN These lines read all branches
      // TBranch *br = branches_->find(poolNames::keyName(*pit))->second;
      // br->SetAddress(p);
      // br->GetEntry(rootFile_->entryNumber());
      // BranchDescription const& product = products_[pit->productID_];
      // std::auto_ptr<Provenance> prov(new Provenance(product, *pit));
      // bool const isPresent = prov->event.isPresent();
      // std::auto_ptr<Group> g(new Group(std::auto_ptr<EDProduct>(p), prov, isPresent));
      // END These lines read all branches
      // BEGIN These lines defer reading branches
      BranchDescription const& product = products_[pit->productID_];
      std::auto_ptr<Provenance> prov(new Provenance(product, *pit));
      bool const isPresent = prov->event.isPresent();
      std::auto_ptr<Group> g(new Group(prov, isPresent));
      // END These lines defer reading branches
      item.addGroup(g);
    }
  }

  boost::shared_ptr<DelayedReader>
  RootTree::makeDelayedReader() const {
    boost::shared_ptr<DelayedReader> store(new RootDelayedReader(entryNumber_, branches_, filePtr_));
    return store;
  }

  RootTree::EntryNumber
  RootTree::getBestEntryNumber(unsigned int major, unsigned int minor) const {
    EntryNumber index = getExactEntryNumber(major, minor);
    if (index < 0) index = tree_->GetEntryNumberWithBestIndex(major, minor) + 1;
    if (index >= entries_) index = -1;
    return index;
  }

  RootTree::EntryNumber
  RootTree::getExactEntryNumber(unsigned int major, unsigned int minor) const {
    EntryNumber index = tree_->GetEntryNumberWithIndex(major, minor);
    if (index < 0) index = -1;
    return index;
  }
}
