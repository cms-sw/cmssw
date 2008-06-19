#include "RootTree.h"
#include "RootDelayedReader.h"
#include "FWCore/Framework/interface/Principal.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ConstBranchDescription.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "Rtypes.h"
#include "TFile.h"
#include "TVirtualIndex.h"
#include "TTreeIndex.h"
#include "TTreeCache.h"
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
    TBranch * getEventEntryInfoBranch(TTree * tree, BranchType const& branchType) {
      TBranch *branch = tree->GetBranch(BranchTypeToBranchEntryInfoBranchName(branchType).c_str());
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
    branchType_(branchType),
    auxBranch_(tree_ ? getAuxiliaryBranch(tree_, branchType_) : 0),
    branchEntryInfoBranch_(metaTree_ ? getEventEntryInfoBranch(metaTree_, branchType_) : 0),
    entries_(tree_ ? tree_->GetEntries() : 0),
    entryNumber_(-1),
    branchNames_(),
    branches_(new BranchMap),
    productStatuses_(),
    pProductStatuses_(&productStatuses_),
    infoTree_(dynamic_cast<TTree *>(filePtr_.get() != 0 ? filePtr->Get(BranchTypeToInfoTreeName(branchType).c_str()) : 0)),
    statusBranch_(infoTree_ ? getStatusBranch(infoTree_, branchType_) : 0)
  { }

  bool
  RootTree::isValid() const {
    if (metaTree_ == 0 || metaTree_->GetNbranches() == 0) {
      return tree_ != 0 && auxBranch_ != 0 && tree_->GetNbranches() == 1; 
    }
    if (tree_ != 0 && auxBranch_ != 0 && metaTree_ != 0) {
      if (branchEntryInfoBranch_ != 0 || statusBranch_ != 0) return true;
      return (entries_ == metaTree_->GetEntries() && tree_->GetNbranches() <= metaTree_->GetNbranches() + 1); 
    }
    return false;
  }

  void
  RootTree::setPresence(BranchDescription const& prod) {
      assert(isValid());
      prod.init();
      prod.setPresent(tree_->GetBranch(prod.branchName().c_str()) != 0);
  }

  void
  RootTree::addBranch(BranchKey const& key,
		      BranchDescription const& prod,
		      std::string const& oldBranchName) {
      assert(isValid());
      prod.init();
      //use the translated branch name 
      TBranch * branch = tree_->GetBranch(oldBranchName.c_str());
      assert (prod.present() == (branch != 0));
      input::BranchInfo info = input::BranchInfo(ConstBranchDescription(prod));
      info.productBranch_ = 0;
      if (prod.present()) {
        info.productBranch_ = branch;
        //we want the new branch name for the JobReport
        branchNames_.push_back(prod.branchName());
      }
      info.provenanceBranch_ = metaTree_->GetBranch(oldBranchName.c_str());
      branches_->insert(std::make_pair(key, info));
  }

  void
  RootTree::dropBranch(std::string const& oldBranchName) {
      //use the translated branch name 
      TBranch * branch = tree_->GetBranch(oldBranchName.c_str());
      if (branch != 0) {
	TObjArray * leaves = tree_->GetListOfLeaves();
	int entries = leaves->GetEntries();
	for (int i = 0; i < entries; ++i) {
	  TLeaf *leaf = (TLeaf *)(*leaves)[i];
	  if (leaf == 0) continue;
	  TBranch* br = leaf->GetBranch();
	  if (br == 0) continue;
	  if (br->GetMother() == branch) {
	    leaves->Remove(leaf);
	  }
	}
	leaves->Compress();
	tree_->GetListOfBranches()->Remove(branch);
	tree_->GetListOfBranches()->Compress();
	delete branch;
      }
  }

  boost::shared_ptr<DelayedReader>
  RootTree::makeDelayedReader() const {
    boost::shared_ptr<DelayedReader> store(new RootDelayedReader(entryNumber_, branches_, filePtr_));
    return store;
  }

  void
  RootTree::setCacheSize(unsigned int cacheSize) const {
    tree_->SetCacheSize(static_cast<Long64_t>(cacheSize));
  }

  void
  RootTree::setTreeMaxVirtualSize(int treeMaxVirtualSize) {
    if (treeMaxVirtualSize >= 0) tree_->SetMaxVirtualSize(static_cast<Long64_t>(treeMaxVirtualSize));
  }

  void
  RootTree::setEntryNumber(EntryNumber theEntryNumber) {
    if (TTreeCache *tc = dynamic_cast<TTreeCache *>(filePtr_->GetCacheRead())) {
      if (theEntryNumber >= 0 && tc->GetOwner() == tree_ && tc->IsLearning()) {
	tc->SetLearnEntries(1);
	tc->SetEntryRange(0, tree_->GetEntries());
        for (BranchMap::const_iterator i = branches_->begin(), e = branches_->end(); i != e; ++i) {
	  if (i->second.productBranch_) {
	    tc->AddBranch(i->second.productBranch_, kTRUE);
	  }
	}
        tc->StopLearningPhase();
      }
    }

    entryNumber_ = theEntryNumber;
    tree_->LoadTree(theEntryNumber);
  }
}
