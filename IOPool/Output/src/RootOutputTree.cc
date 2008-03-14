#include "RootOutputTree.h"
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TTreeCloner.h"
#include "TBranchElement.h"
#include "TStreamerInfo.h"
#include "FWCore/Utilities/interface/Algorithms.h"

#include "boost/bind.hpp"
#include <algorithm>
#include <limits>
#include <cstring>

namespace edm {
  TTree *
  RootOutputTree::assignTTree(TFile * filePtr, TTree * tree) {
    tree->SetDirectory(filePtr);
    // Turn off autosaving because it is such a memory hog and we are not using
    // this check-pointing feature anyway.
    tree->SetAutoSave(std::numeric_limits<Long64_t>::max());
    return tree;
  }

  TTree *
  RootOutputTree::makeTTree(TFile * filePtr, std::string const& name, int splitLevel) {
    TTree *tree = new TTree(name.c_str(), "", splitLevel);
    return assignTTree(filePtr, tree);
  }

  void
  RootOutputTree::fastCloneTTree(TTree *in, TTree *out) {
    TTreeCloner cloner(in, out, "");
    if (!cloner.IsValid()) {
       throw edm::Exception(edm::errors::FatalRootError)
         << "invalid TTreeCloner\n";
    }
    out->SetEntries(out->GetEntries() + in->GetEntries());
    cloner.Exec();
  }
 
  void
  RootOutputTree::writeTTree(TTree *tree) {
    if (tree->GetNbranches() != 0) {
      tree->SetEntries(-1);
    }
    tree->AutoSave();
  }

  void
  RootOutputTree::fillTTree(TTree * tree, std::vector<TBranch *> const& branches) {
    for_all(branches, boost::bind(&TBranch::Fill, _1));
  }

  void
  RootOutputTree::writeTree() const {
    writeTTree(tree_);
    writeTTree(metaTree_);
  }

  void
  RootOutputTree::fastCloneTree(TTree *tree, TTree *metaTree) {
    if (currentlyFastCloning_) {
      fastCloneTTree(metaTree, metaTree_);
      fastCloneTTree(tree, tree_);
    }
  }

  void
  RootOutputTree::fillTree() const {
    fillTTree(metaTree_, metaBranches_);
    fillTTree(tree_, branches_);
    if (!currentlyFastCloning_) {
      fillTTree(metaTree_, clonedMetaBranches_);
      fillTTree(tree_, clonedBranches_);
    }
  }

  void
  RootOutputTree::addBranch(BranchDescription const& prod,
			    bool selected,
			    BranchEntryDescription const*& pProv,
			    void const*& pProd, bool inInput) {
      prod.init();
      TBranch *meta = metaTree_->Branch(prod.branchName().c_str(), &pProv, basketSize_, 0);
      if (inInput) {
	clonedMetaBranches_.push_back(meta);
      } else {
	metaBranches_.push_back(meta);
      }
      if (selected) {
	TBranch *branch = tree_->Branch(prod.branchName().c_str(),
		 prod.wrappedName().c_str(),
		 &pProd,
		 (prod.basketSize() == BranchDescription::invalidBasketSize ? basketSize_ : prod.basketSize()),
		 (prod.splitLevel() == BranchDescription::invalidSplitLevel ? splitLevel_ : prod.splitLevel()));
        if (inInput) {
	  clonedBranches_.push_back(branch);
	} else {
	  branches_.push_back(branch);
	}
	// we want the new branch name for the JobReport
	branchNames_.push_back(prod.branchName());
      }
  }
}
