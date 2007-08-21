#include "RootOutputTree.h"
#include "TFile.h"

namespace edm {

  TTree *
  RootOutputTree::makeTree(TFile * filePtr, std::string const& name, int splitLevel) {
    TTree *tree = new TTree(name.c_str(), "", splitLevel);
    tree->SetDirectory(filePtr);
    return tree;
  }

  void
  RootOutputTree::addBranch(BranchDescription const& prod, bool selected, BranchEntryDescription const*& pProv, void const*& pProd) {
      prod.init();
      metaTree_->Branch(prod.branchName().c_str(), &pProv, basketSize_, 0);
      if (selected) {
	tree_->Branch(prod.branchName().c_str(),
		       wrappedClassName(prod.className()).c_str(),
		       &pProd,
		       basketSize_,
		       splitLevel_);
	// we want the new branch name for the JobReport
	branchNames_.push_back(prod.branchName());
      }
  }
}
