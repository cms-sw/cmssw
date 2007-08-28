#include "RootOutputTree.h"
#include "TFile.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"

namespace edm {

  TTree *
  RootOutputTree::makeTree(TFile * filePtr, std::string const& name, int splitLevel) {
    TTree *tree = new TTree(name.c_str(), "", splitLevel);
    tree->SetDirectory(filePtr);
    return tree;
  }

  void
  RootOutputTree::writeTTree(TTree *tree) {
    tree->AutoSave();
  }

  void
  RootOutputTree::writeTree() const {
    writeTTree(tree_);
    writeTTree(metaTree_);
  }

  void
  RootOutputTree::addBranch(BranchDescription const& prod, bool selected, BranchEntryDescription const*& pProv, void const*& pProd) {
      prod.init();
      metaTree_->Branch(prod.branchName().c_str(), &pProv, basketSize_, 0);
      if (selected) {
	tree_->Branch(prod.branchName().c_str(),
		       wrappedClassName(prod.className()).c_str(),
		       &pProd,
		       (prod.basketSize() == BranchDescription::invalidBasketSize ? basketSize_ : prod.basketSize()),
		       (prod.splitLevel() == BranchDescription::invalidSplitLevel ? splitLevel_ : prod.splitLevel()));
	// we want the new branch name for the JobReport
	branchNames_.push_back(prod.branchName());
      }
  }
}
