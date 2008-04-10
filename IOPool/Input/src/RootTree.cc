#include "RootTree.h"
#include "RootDelayedReader.h"
#include "FWCore/Framework/interface/Principal.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"
#include "DataFormats/Provenance/interface/BranchKey.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ConstBranchDescription.h"
#include "Rtypes.h"
#include "TTree.h"
#include "TFile.h"
#include "TVirtualIndex.h"
#include "TTreeIndex.h"
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
    TBranch * getStatusBranch(TTree * tree, BranchType const& branchType) {
      TBranch *branch = tree->GetBranch(BranchTypeToProductStatusBranchName(branchType).c_str());
      return branch;
    }
  }
  RootTree::RootTree(boost::shared_ptr<TFile> filePtr, BranchType const& branchType) :
    filePtr_(filePtr),
    tree_(dynamic_cast<TTree *>(filePtr_.get() != 0 ? filePtr->Get(BranchTypeToProductTreeName(branchType).c_str()) : 0)),
    metaTree_(dynamic_cast<TTree *>(filePtr_.get() != 0 ? filePtr->Get(BranchTypeToMetaDataTreeName(branchType).c_str()) : 0)),
    infoTree_(dynamic_cast<TTree *>(filePtr_.get() != 0 ? filePtr->Get(BranchTypeToInfoTreeName(branchType).c_str()) : 0)),
    branchType_(branchType),
    auxBranch_(tree_ ? getAuxiliaryBranch(tree_, branchType_) : 0),
    statusBranch_(infoTree_ ? getStatusBranch(infoTree_, branchType_) : 0),
    entries_(tree_ ? tree_->GetEntries() : 0),
    entryNumber_(-1),
    branchNames_(),
    branches_(new BranchMap),
    productStatuses_(),
    pProductStatuses_(&productStatuses_)
  { }

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
  RootTree::setPresence(
		      BranchDescription const& prod) {
      assert(isValid());
      prod.init();
      prod.provenancePresent_ = (metaTree_->GetBranch(prod.branchName().c_str()) != 0);
      prod.present_ = (tree_->GetBranch(prod.branchName().c_str()) != 0);
  }

  void
  RootTree::addBranch(BranchKey const& key,
		      BranchDescription const& prod,
		      std::string const& oldBranchName) {
      assert(isValid());
      prod.init();
      //use the translated branch name 
      TBranch * provBranch = metaTree_->GetBranch(oldBranchName.c_str());
      assert (prod.provenancePresent_ == (provBranch != 0));
      TBranch * branch = tree_->GetBranch(oldBranchName.c_str());
      assert (prod.present_ == (branch != 0));
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
    if (productStatuses_.empty()) {
      // For backward compatibility
      for (; pit != pitEnd; ++pit) {
        ConstBranchDescription const& bd = pit->second.branchDescription_;
        item.addGroup(bd, productstatus::unknown());
      }
    } else {
      for (; pit != pitEnd; ++pit) {
        ConstBranchDescription const& bd = pit->second.branchDescription_;
        item.addGroup(bd, productStatuses_[bd.productID().id() - 1]);
      }
    }
  }

  boost::shared_ptr<DelayedReader>
  RootTree::makeDelayedReader(FileFormatVersion const& fileFormatVersion) const {
    boost::shared_ptr<DelayedReader> store(new RootDelayedReader(entryNumber_, branches_, filePtr_, fileFormatVersion));
    return store;
  }

  void
  RootTree::setCacheSize(unsigned int cacheSize) const {
    tree_->SetCacheSize(static_cast<Long64_t>(cacheSize));
  }
}
