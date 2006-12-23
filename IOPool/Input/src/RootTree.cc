#include "IOPool/Input/src/RootTree.h"

#include "TFile.h"
#include "TTree.h"

#include <iostream>

namespace edm {
  RootTree::RootTree(TFile& file, BranchType const& branchType) :
  tree_(dynamic_cast<TTree *>(file.Get(BranchTypeToProductTreeName(branchType).c_str()))),
  metaTree_(dynamic_cast<TTree *>(file.Get(BranchTypeToMetaDataTreeName(branchType).c_str()))),
  auxBranch_(tree_ ? tree_->GetBranch(BranchTypeToAuxiliaryBranchName(branchType).c_str()): 0),
  entries_(tree_ ? tree_->GetEntries() : 0),
  entryNumber_(-1),
  origEntryNumber_(),
  branchNames_(),
  provenance_(),
  provenancePtrs_()
  {
    provenance_.reserve(1000);
    provenancePtrs_.reserve(1000);
    if (tree_ != 0 && metaTree_ != 0) {
      assert(entries_ == metaTree_->GetEntries());
      assert(auxBranch_ != 0);
    }
    else {
      // For backward compatibility
      tree_ = 0;
      metaTree_ = 0;
      auxBranch_ = 0;
      entries_ = 0;
    }
  }

  void
  RootTree::addBranch(BranchKey const& key,
		      BranchDescription const& prod,
		      BranchMap & branches,
		      ProductMap & products,
		      std::string const& oldBranchName) {
      prod.init();
      //use the translated branch name 
      prod.provenancePresent_ = (metaTree_->GetBranch(oldBranchName.c_str()) != 0);
      TBranch * branch = tree_->GetBranch(oldBranchName.c_str());
      prod.present_ = (branch != 0);
      if (prod.provenancePresent()) {
        std::string const &name = prod.className();
        std::string const className = wrappedClassName(name);
        if (branch != 0) branches.insert(std::make_pair(key, std::make_pair(className, branch)));
        products.insert(std::make_pair(prod.productID(), prod));
	//we want the new branch name for the JobReport
	branchNames_.push_back(prod.branchName());
        int n = provenance_.size();
        provenance_.push_back(BranchEntryDescription());
        provenancePtrs_.push_back(&provenance_[n]);
        metaTree_->SetBranchAddress(oldBranchName.c_str(),(&provenancePtrs_[n]));
      }
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
