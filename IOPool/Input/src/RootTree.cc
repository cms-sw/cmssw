#include "RootTree.h"
#include "RootDelayedReader.h"
#include "FWCore/Framework/interface/Principal.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"
#include "DataFormats/Provenance/interface/BranchKey.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ConstBranchDescription.h"
#include "TTree.h"
#include "TFile.h"
class TBranch;

#include <iostream>

namespace edm {
  namespace {
    TBranch * getAuxiliaryBranch(TTree * tree, BranchType const& branchType) {
      TBranch *branch = tree->GetBranch(BranchTypeToAuxiliaryBranchName(branchType).c_str());
      if (branch == 0) {
        branch = tree->GetBranch(BranchTypeToAuxBranchName(branchType).c_str());
      }
      return branch;
    }
  }
  RootTree::RootTree(boost::shared_ptr<TFile> filePtr, BranchType const& branchType) :
    filePtr_(filePtr),
    tree_(dynamic_cast<TTree *>(filePtr_.get() != 0 ? filePtr->Get(BranchTypeToProductTreeName(branchType).c_str()) : 0)),
    metaTree_(dynamic_cast<TTree *>(filePtr_.get() != 0 ? filePtr->Get(BranchTypeToMetaDataTreeName(branchType).c_str()) : 0)),

    auxBranch_(tree_ ? getAuxiliaryBranch(tree_, branchType) : 0),
    entries_(tree_ ? tree_->GetEntries() : 0),
    entryNumber_(-1),
    origEntryNumber_(),
    branchNames_(),
    branches_(new BranchMap)
  {}

  bool
  RootTree::isValid() const {
    if (metaTree_ == 0 || metaTree_->GetNbranches() == 0) {
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
      assert(isValid());
      prod.init();
      //use the translated branch name 
      TBranch * provBranch = metaTree_->GetBranch(oldBranchName.c_str());
      prod.provenancePresent_ = (metaTree_->GetBranch(oldBranchName.c_str()) != 0);
      TBranch * branch = tree_->GetBranch(oldBranchName.c_str());
      prod.present_ = (branch != 0);
      if (prod.provenancePresent()) {
        input::EventBranchInfo info = input::EventBranchInfo(ConstBranchDescription(prod));
        info.provenanceBranch_ = provBranch;
        info.productBranch_ = 0;
	if (prod.present_) {
          info.productBranch_ = branch;
	  //we want the new branch name for the JobReport
	  branchNames_.push_back(prod.branchName());
        }
	branches_->insert(std::make_pair(key, info));
      }
  }

  void
  RootTree::fillGroups(Principal& item) {
    if (metaTree_ == 0 || metaTree_->GetNbranches() == 0) return;
    // Loop over provenance
    BranchMap::const_iterator pit = branches_->begin(), pitEnd = branches_->end();
    for (; pit != pitEnd; ++pit) {
      item.addGroup(pit->second.branchDescription_);
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
